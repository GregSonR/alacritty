//! Terminal window context.

use std::error::Error;
use std::fs::File;
use std::io::Write;
use std::mem;
#[cfg(not(windows))]
use std::os::unix::io::{AsRawFd, RawFd};
use std::rc::Rc;
use std::sync::Arc;
use std::time::Instant;

use glutin::config::Config as GlutinConfig;
use glutin::display::GetGlDisplay;
#[cfg(all(feature = "x11", not(any(target_os = "macos", windows))))]
use glutin::platform::x11::X11GlConfigExt;
use log::info;
use serde_json as json;
use winit::event::{Event as WinitEvent, Modifiers, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoopProxy};
use winit::raw_window_handle::HasDisplayHandle;
use winit::window::WindowId;

use alacritty_terminal::event::{Event as TerminalEvent, OnResize};
use alacritty_terminal::event_loop::{EventLoop as PtyEventLoop, Msg, Notifier};
use alacritty_terminal::grid::{Dimensions, Scroll};
use alacritty_terminal::index::Direction;
use alacritty_terminal::sync::FairMutex;
use alacritty_terminal::term::test::TermSize;
use alacritty_terminal::term::{Term, TermMode};
use alacritty_terminal::tty;

use crate::cli::{ParsedOptions, WindowOptions};
use crate::clipboard::Clipboard;
use crate::config::UiConfig;
#[cfg(not(windows))]
use crate::daemon::foreground_process_path;
use crate::display::{Display, TabBarHit};
use crate::display::window::Window;
use crate::event::{
    ActionContext, Event, EventProxy, InlineSearchState, Mouse, SearchState, TouchPurpose,
};
#[cfg(unix)]
use crate::logging::LOG_TARGET_IPC_CONFIG;
use crate::message_bar::MessageBuffer;
use crate::scheduler::Scheduler;
use crate::{input, renderer};

/// Event context for one individual Alacritty window.
pub struct WindowContext {
    pub message_buffer: MessageBuffer,
    pub display: Display,
    pub dirty: bool,
    event_queue: Vec<WinitEvent<Event>>,
    tabs: Vec<TerminalTab>,
    active_tab: usize,
    next_tab_id: u64,
    cursor_blink_timed_out: bool,
    prev_bell_cmd: Option<Instant>,
    modifiers: Modifiers,
    inline_search_state: InlineSearchState,
    search_state: SearchState,
    mouse: Mouse,
    touch: TouchPurpose,
    occluded: bool,
    window_config: ParsedOptions,
    config: Rc<UiConfig>,
}

struct TerminalTab {
    id: u64,
    title: String,
    terminal: Arc<FairMutex<Term<EventProxy>>>,
    notifier: Notifier,
    preserve_title: bool,
    #[cfg(not(windows))]
    master_fd: RawFd,
    #[cfg(not(windows))]
    shell_pid: u32,
}

impl WindowContext {
    /// Create initial window context that does bootstrapping the graphics API we're going to use.
    pub fn initial(
        event_loop: &ActiveEventLoop,
        proxy: EventLoopProxy<Event>,
        config: Rc<UiConfig>,
        mut options: WindowOptions,
    ) -> Result<Self, Box<dyn Error>> {
        let raw_display_handle = event_loop.display_handle().unwrap().as_raw();

        let mut identity = config.window.identity.clone();
        options.window_identity.override_identity_config(&mut identity);

        // Windows has different order of GL platform initialization compared to any other platform;
        // it requires the window first.
        #[cfg(windows)]
        let window = Window::new(event_loop, &config, &identity, &mut options)?;
        #[cfg(windows)]
        let raw_window_handle = Some(window.raw_window_handle());

        #[cfg(not(windows))]
        let raw_window_handle = None;

        let gl_display = renderer::platform::create_gl_display(
            raw_display_handle,
            raw_window_handle,
            config.debug.prefer_egl,
        )?;
        let gl_config = renderer::platform::pick_gl_config(&gl_display, raw_window_handle)?;

        #[cfg(not(windows))]
        let window = Window::new(
            event_loop,
            &config,
            &identity,
            &mut options,
            #[cfg(all(feature = "x11", not(any(target_os = "macos", windows))))]
            gl_config.x11_visual(),
        )?;

        // Create context.
        let gl_context =
            renderer::platform::create_gl_context(&gl_display, &gl_config, raw_window_handle)?;

        let display = Display::new(window, gl_context, &config, false)?;

        Self::new(display, config, options, proxy)
    }

    /// Create additional context with the graphics platform other windows are using.
    pub fn additional(
        gl_config: &GlutinConfig,
        event_loop: &ActiveEventLoop,
        proxy: EventLoopProxy<Event>,
        config: Rc<UiConfig>,
        mut options: WindowOptions,
        config_overrides: ParsedOptions,
    ) -> Result<Self, Box<dyn Error>> {
        let gl_display = gl_config.display();

        let mut identity = config.window.identity.clone();
        options.window_identity.override_identity_config(&mut identity);

        // Check if new window will be opened as a tab.
        // This must be done before `Window::new()`, which unsets `window_tabbing_id`.
        #[cfg(target_os = "macos")]
        let tabbed = options.window_tabbing_id.is_some();
        #[cfg(not(target_os = "macos"))]
        let tabbed = false;

        let window = Window::new(
            event_loop,
            &config,
            &identity,
            &mut options,
            #[cfg(all(feature = "x11", not(any(target_os = "macos", windows))))]
            gl_config.x11_visual(),
        )?;

        // Create context.
        let raw_window_handle = window.raw_window_handle();
        let gl_context =
            renderer::platform::create_gl_context(&gl_display, gl_config, Some(raw_window_handle))?;

        let display = Display::new(window, gl_context, &config, tabbed)?;

        let mut window_context = Self::new(display, config, options, proxy)?;

        // Set the config overrides at startup.
        //
        // These are already applied to `config`, so no update is necessary.
        window_context.window_config = config_overrides;

        Ok(window_context)
    }

    /// Create a new terminal window context.
    fn new(
        display: Display,
        config: Rc<UiConfig>,
        options: WindowOptions,
        proxy: EventLoopProxy<Event>,
    ) -> Result<Self, Box<dyn Error>> {
        let tab = Self::create_terminal_tab(&display, &config, options, proxy, 0)?;

        // Create context for the Alacritty window.
        Ok(WindowContext {
            tabs: vec![tab],
            active_tab: 0,
            next_tab_id: 1,
            display,
            config,
            cursor_blink_timed_out: Default::default(),
            prev_bell_cmd: Default::default(),
            inline_search_state: Default::default(),
            message_buffer: Default::default(),
            window_config: Default::default(),
            search_state: Default::default(),
            event_queue: Default::default(),
            modifiers: Default::default(),
            occluded: Default::default(),
            mouse: Default::default(),
            touch: Default::default(),
            dirty: Default::default(),
        })
    }

    fn create_terminal_tab(
        display: &Display,
        config: &Rc<UiConfig>,
        options: WindowOptions,
        proxy: EventLoopProxy<Event>,
        tab_id: u64,
    ) -> Result<TerminalTab, Box<dyn Error>> {
        let preserve_title = options.window_identity.title.is_some();
        let mut pty_config = config.pty_config();
        options.terminal_options.override_pty_config(&mut pty_config);

        info!(
            "PTY dimensions: {:?} x {:?}",
            display.size_info.screen_lines(),
            display.size_info.columns()
        );

        let event_proxy = EventProxy::new(proxy, display.window.id(), tab_id);

        // Create the terminal.
        //
        // This object contains all of the state about what's being displayed. It's
        // wrapped in a clonable mutex since both the I/O loop and display need to
        // access it.
        let terminal = Term::new(config.term_options(), &display.size_info, event_proxy.clone());
        let terminal = Arc::new(FairMutex::new(terminal));

        // Create the PTY.
        //
        // The PTY forks a process to run the shell on the slave side of the
        // pseudoterminal. A file descriptor for the master side is retained for
        // reading/writing to the shell.
        let pty = tty::new(&pty_config, display.size_info.into(), display.window.id().into())?;

        #[cfg(not(windows))]
        let master_fd = pty.file().as_raw_fd();
        #[cfg(not(windows))]
        let shell_pid = pty.child().id();

        // Create the pseudoterminal I/O loop.
        //
        // PTY I/O is ran on another thread as to not occupy cycles used by the
        // renderer and input processing. Note that access to the terminal state is
        // synchronized since the I/O loop updates the state, and the display
        // consumes it periodically.
        let event_loop = PtyEventLoop::new(
            Arc::clone(&terminal),
            event_proxy.clone(),
            pty,
            pty_config.drain_on_exit,
            config.debug.ref_test,
        )?;

        // The event loop channel allows write requests from the event processor
        // to be sent to the pty loop and ultimately written to the pty.
        let loop_tx = event_loop.channel();

        // Kick off the I/O thread.
        let _io_thread = event_loop.spawn();

        // Start cursor blinking, in case `Focused` isn't sent on startup.
        if config.cursor.style().blinking {
            event_proxy.send_event(TerminalEvent::CursorBlinkingChange.into());
        }

        Ok(TerminalTab {
            id: tab_id,
            title: options.window_identity.title.unwrap_or_else(|| {
                config.window.identity.title.clone()
            }),
            preserve_title,
            terminal,
            #[cfg(not(windows))]
            master_fd,
            #[cfg(not(windows))]
            shell_pid,
            notifier: Notifier(loop_tx),
        })
    }

    /// Update the terminal window to the latest config.
    pub fn update_config(&mut self, new_config: Rc<UiConfig>) {
        let old_config = mem::replace(&mut self.config, new_config);

        // Apply ipc config if there are overrides.
        self.config = self.window_config.override_config_rc(self.config.clone());

        self.display.update_config(&self.config);
        for tab in &mut self.tabs {
            tab.terminal.lock().set_options(self.config.term_options());
        }

        // Reload cursor if its thickness has changed.
        if (old_config.cursor.thickness() - self.config.cursor.thickness()).abs() > f32::EPSILON {
            self.display.pending_update.set_cursor_dirty();
        }

        if old_config.font != self.config.font {
            let scale_factor = self.display.window.scale_factor as f32;
            // Do not update font size if it has been changed at runtime.
            if self.display.font_size == old_config.font.size().scale(scale_factor) {
                self.display.font_size = self.config.font.size().scale(scale_factor);
            }

            let font = self.config.font.clone().with_size(self.display.font_size);
            self.display.pending_update.set_font(font);
        }

        // Always reload the theme to account for auto-theme switching.
        self.display.window.set_theme(self.config.window.theme());

        // Update display if either padding options or resize increments were changed.
        let window_config = &old_config.window;
        if window_config.padding(1.) != self.config.window.padding(1.)
            || window_config.dynamic_padding != self.config.window.dynamic_padding
            || window_config.resize_increments != self.config.window.resize_increments
        {
            self.display.pending_update.dirty = true;
        }

        // Update title on config reload according to the following table.
        //
        // │cli │ dynamic_title │ current_title == old_config ││ set_title │
        // │ Y  │       _       │              _              ││     N     │
        // │ N  │       Y       │              Y              ││     Y     │
        // │ N  │       Y       │              N              ││     N     │
        // │ N  │       N       │              _              ││     Y     │
        if !self.tabs[self.active_tab].preserve_title
            && (!self.config.window.dynamic_title
                || self.display.window.title() == old_config.window.identity.title)
        {
            self.display.window.set_title(self.config.window.identity.title.clone());
        }

        let opaque = self.config.window_opacity() >= 1.;

        // Disable shadows for transparent windows on macOS.
        #[cfg(target_os = "macos")]
        self.display.window.set_has_shadow(opaque);

        #[cfg(target_os = "macos")]
        self.display.window.set_option_as_alt(self.config.window.option_as_alt());

        // Change opacity and blur state.
        self.display.window.set_transparent(!opaque);
        self.display.window.set_blur(self.config.window.blur);

        // Update hint keys.
        self.display.hint_state.update_alphabet(self.config.hints.alphabet());

        // Update cursor blinking.
        let event = Event::new(TerminalEvent::CursorBlinkingChange.into(), None);
        self.event_queue.push(event.into());

        self.dirty = true;
    }

    /// Get reference to the window's configuration.
    #[cfg(unix)]
    pub fn config(&self) -> &UiConfig {
        &self.config
    }

    /// Clear the window config overrides.
    #[cfg(unix)]
    pub fn reset_window_config(&mut self, config: Rc<UiConfig>) {
        // Clear previous window errors.
        self.message_buffer.remove_target(LOG_TARGET_IPC_CONFIG);

        self.window_config.clear();

        // Reload current config to pull new IPC config.
        self.update_config(config);
    }

    /// Add new window config overrides.
    #[cfg(unix)]
    pub fn add_window_config(&mut self, config: Rc<UiConfig>, options: &ParsedOptions) {
        // Clear previous window errors.
        self.message_buffer.remove_target(LOG_TARGET_IPC_CONFIG);

        self.window_config.extend_from_slice(options);

        // Reload current config to pull new IPC config.
        self.update_config(config);
    }

    pub fn create_tab(&mut self, proxy: EventLoopProxy<Event>) -> Result<(), Box<dyn Error>> {
        let mut options = WindowOptions::default();

        #[cfg(not(windows))]
        {
            let active = &self.tabs[self.active_tab];
            options.terminal_options.working_directory =
                foreground_process_path(active.master_fd, active.shell_pid).ok();
        }

        let tab = Self::create_terminal_tab(
            &self.display,
            &self.config,
            options,
            proxy,
            self.next_tab_id,
        )?;
        self.next_tab_id += 1;
        self.tabs.push(tab);
        self.active_tab = self.tabs.len() - 1;
        self.sync_active_tab();
        Ok(())
    }

    pub fn select_tab(&mut self, index: usize) {
        if index >= self.tabs.len() || index == self.active_tab {
            return;
        }

        self.active_tab = index;
        self.sync_active_tab();
    }

    pub fn close_tab(&mut self, index: usize) {
        if index >= self.tabs.len() {
            return;
        }

        if self.tabs.len() == 1 {
            self.tabs[index].terminal.lock().exit();
            return;
        }

        let tab = self.tabs.remove(index);
        let _ = tab.notifier.0.send(Msg::Shutdown);
        if self.active_tab >= self.tabs.len() {
            self.active_tab = self.tabs.len() - 1;
        } else if index < self.active_tab {
            self.active_tab -= 1;
        }

        self.sync_active_tab();
    }

    pub fn handle_tab_terminal_event(&mut self, tab_id: u64, event: TerminalEvent) {
        let Some(index) = self.tabs.iter().position(|tab| tab.id == tab_id) else {
            return;
        };

        match event {
            TerminalEvent::Exit => self.close_tab(index),
            TerminalEvent::Title(title) => {
                self.tabs[index].title = title.clone();
                self.update_tab_bar();
                self.dirty = true;
                if index == self.active_tab {
                    self.event_queue
                        .push(Event::new(TerminalEvent::Title(title).into(), self.id()).into());
                }
            },
            TerminalEvent::ResetTitle => {
                self.tabs[index].title = self.config.window.identity.title.clone();
                self.update_tab_bar();
                self.dirty = true;
                if index == self.active_tab {
                    self.event_queue
                        .push(Event::new(TerminalEvent::ResetTitle.into(), self.id()).into());
                }
            },
            event => {
                if index == self.active_tab {
                    // Mark the window dirty so the active terminal is redrawn. This is
                    // essential for `Wakeup`, which signals new PTY output but is a no-op
                    // in the input processor; without it the screen only refreshes on
                    // incidental events (cursor blink, keypresses), making the terminal
                    // appear laggy and swallow typed input.
                    self.dirty = true;
                    self.event_queue.push(Event::new(event.into(), self.id()).into());
                }
            },
        }
    }

    fn update_tab_bar(&mut self) {
        let titles = self.tabs.iter().map(|tab| tab.title.clone()).collect();
        self.display.set_tab_bar(titles, self.active_tab);
    }

    /// Synchronise the display and the active terminal after a tab change.
    ///
    /// Creating, selecting or closing a tab can toggle the tab bar's visibility
    /// (which steals/returns a grid line), and the terminal that is about to be
    /// displayed may still carry a grid size from when it was last active. Both
    /// must be reconciled here, since the redraw path does not run the regular
    /// display-update logic that resizes the grid on window events.
    fn sync_active_tab(&mut self) {
        let tab_bar_visible = self.tabs.len() > 1;
        let old_is_searching = self.search_state.history_index.is_some();

        self.update_tab_bar();

        // Recompute `size_info`, accounting for the tab bar appearing/disappearing.
        // This resizes the active terminal when the available grid size changes.
        self.display.pending_update.dirty = true;
        {
            let active_tab = &mut self.tabs[self.active_tab];
            let mut terminal = active_tab.terminal.lock();
            Self::submit_display_update(
                &mut terminal,
                &mut self.display,
                &mut active_tab.notifier,
                &self.message_buffer,
                &mut self.search_state,
                old_is_searching,
                tab_bar_visible,
                &self.config,
            );
        }

        // `submit_display_update` only resizes the terminal when `size_info`
        // itself changed. When switching between tabs while the bar stays
        // visible the size is unchanged, yet the newly active terminal can hold
        // an outdated grid size, so force it to match the current display.
        let size_info = self.display.size_info;
        let active_tab = &mut self.tabs[self.active_tab];
        let mut terminal = active_tab.terminal.lock();
        if terminal.screen_lines() != size_info.screen_lines()
            || terminal.columns() != size_info.columns()
        {
            terminal.resize(size_info);
            active_tab.notifier.on_resize(size_info.into());
        }
        drop(terminal);

        self.dirty = true;
        self.request_redraw();
    }

    fn request_redraw(&mut self) {
        if self.display.window.has_frame && !self.occluded {
            self.display.window.request_redraw();
        }
    }

    /// Draw the window.
    pub fn draw(&mut self, scheduler: &mut Scheduler) {
        self.display.window.requested_redraw = false;

        if self.occluded {
            return;
        }

        self.dirty = false;

        // Force the display to process any pending display update.
        self.display.process_renderer_update();

        // Request immediate re-draw if visual bell animation is not finished yet.
        if !self.display.visual_bell.completed() {
            // We can get an OS redraw which bypasses alacritty's frame throttling, thus
            // marking the window as dirty when we don't have frame yet.
            if self.display.window.has_frame {
                self.display.window.request_redraw();
            } else {
                self.dirty = true;
            }
        }

        // Redraw the window.
        self.update_tab_bar();
        let terminal = self.tabs[self.active_tab].terminal.lock();
        self.display.draw(
            terminal,
            scheduler,
            &self.message_buffer,
            &self.config,
            &mut self.search_state,
        );
    }

    /// Process events for this terminal window.
    pub fn handle_event(
        &mut self,
        #[cfg(target_os = "macos")] event_loop: &ActiveEventLoop,
        event_proxy: &EventLoopProxy<Event>,
        clipboard: &mut Clipboard,
        scheduler: &mut Scheduler,
        event: WinitEvent<Event>,
    ) {
        match event {
            WinitEvent::AboutToWait
            | WinitEvent::WindowEvent { event: WindowEvent::RedrawRequested, .. } => {
                // Skip further event handling with no staged updates.
                if self.event_queue.is_empty() {
                    return;
                }

                // Continue to process all pending events.
            },
            event => {
                self.event_queue.push(event);
                return;
            },
        }

        if let WinitEvent::WindowEvent {
            event: WindowEvent::MouseInput { state, button: winit::event::MouseButton::Left, .. },
            ..
        } = &event
        {
            if *state == winit::event::ElementState::Pressed {
                if let Some(hit) =
                    self.display.tab_bar_hit(self.mouse.x as f64, self.mouse.y as f64)
                {
                    match hit {
                        TabBarHit::Select(index) => self.select_tab(index),
                        TabBarHit::Close(index) => self.close_tab(index),
                    }
                    return;
                }
            }
        }

        let tab_bar_visible = self.tabs.len() > 1;
        let active_tab = &mut self.tabs[self.active_tab];
        let mut terminal = active_tab.terminal.lock();

        let old_is_searching = self.search_state.history_index.is_some();

        let context = ActionContext {
            cursor_blink_timed_out: &mut self.cursor_blink_timed_out,
            prev_bell_cmd: &mut self.prev_bell_cmd,
            message_buffer: &mut self.message_buffer,
            inline_search_state: &mut self.inline_search_state,
            search_state: &mut self.search_state,
            modifiers: &mut self.modifiers,
            notifier: &mut active_tab.notifier,
            display: &mut self.display,
            mouse: &mut self.mouse,
            touch: &mut self.touch,
            dirty: &mut self.dirty,
            occluded: &mut self.occluded,
            terminal: &mut terminal,
            #[cfg(not(windows))]
            master_fd: active_tab.master_fd,
            #[cfg(not(windows))]
            shell_pid: active_tab.shell_pid,
            preserve_title: active_tab.preserve_title,
            config: &self.config,
            event_proxy,
            #[cfg(target_os = "macos")]
            event_loop,
            clipboard,
            scheduler,
        };
        let mut processor = input::Processor::new(context);

        for event in self.event_queue.drain(..) {
            processor.handle_event(event);
        }

        // Process DisplayUpdate events.
        if self.display.pending_update.dirty {
            Self::submit_display_update(
                &mut terminal,
                &mut self.display,
                &mut active_tab.notifier,
                &self.message_buffer,
                &mut self.search_state,
                old_is_searching,
                tab_bar_visible,
                &self.config,
            );
            self.dirty = true;
        }

        if self.dirty || self.mouse.hint_highlight_dirty {
            self.dirty |= self.display.update_highlighted_hints(
                &terminal,
                &self.config,
                &self.mouse,
                self.modifiers.state(),
            );
            self.mouse.hint_highlight_dirty = false;
        }

        // Don't call `request_redraw` when event is `RedrawRequested` since the `dirty` flag
        // represents the current frame, but redraw is for the next frame.
        if self.dirty
            && self.display.window.has_frame
            && !self.occluded
            && !matches!(event, WinitEvent::WindowEvent { event: WindowEvent::RedrawRequested, .. })
        {
            self.display.window.request_redraw();
        }
    }

    /// ID of this terminal context.
    pub fn id(&self) -> WindowId {
        self.display.window.id()
    }

    /// Write the ref test results to the disk.
    pub fn write_ref_test_results(&self) {
        // Dump grid state.
        let mut grid = self.tabs[self.active_tab].terminal.lock().grid().clone();
        grid.initialize_all();
        grid.truncate();

        let serialized_grid = json::to_string(&grid).expect("serialize grid");

        let size_info = &self.display.size_info;
        let size = TermSize::new(size_info.columns(), size_info.screen_lines());
        let serialized_size = json::to_string(&size).expect("serialize size");

        let serialized_config = format!("{{\"history_size\":{}}}", grid.history_size());

        File::create("./grid.json")
            .and_then(|mut f| f.write_all(serialized_grid.as_bytes()))
            .expect("write grid.json");

        File::create("./size.json")
            .and_then(|mut f| f.write_all(serialized_size.as_bytes()))
            .expect("write size.json");

        File::create("./config.json")
            .and_then(|mut f| f.write_all(serialized_config.as_bytes()))
            .expect("write config.json");
    }

    /// Submit the pending changes to the `Display`.
    fn submit_display_update(
        terminal: &mut Term<EventProxy>,
        display: &mut Display,
        notifier: &mut Notifier,
        message_buffer: &MessageBuffer,
        search_state: &mut SearchState,
        old_is_searching: bool,
        tab_bar_visible: bool,
        config: &UiConfig,
    ) {
        // Compute cursor positions before resize.
        let num_lines = terminal.screen_lines();
        let cursor_at_bottom = terminal.grid().cursor.point.line + 1 == num_lines;
        let origin_at_bottom = if terminal.mode().contains(TermMode::VI) {
            terminal.vi_mode_cursor.point.line == num_lines - 1
        } else {
            search_state.direction == Direction::Left
        };

        display.handle_update(
            terminal,
            notifier,
            message_buffer,
            search_state,
            tab_bar_visible,
            config,
        );

        let new_is_searching = search_state.history_index.is_some();
        if !old_is_searching && new_is_searching {
            // Scroll on search start to make sure origin is visible with minimal viewport motion.
            let display_offset = terminal.grid().display_offset();
            if display_offset == 0 && cursor_at_bottom && !origin_at_bottom {
                terminal.scroll_display(Scroll::Delta(1));
            } else if display_offset != 0 && origin_at_bottom {
                terminal.scroll_display(Scroll::Delta(-1));
            }
        }
    }
}

impl Drop for WindowContext {
    fn drop(&mut self) {
        // Shutdown the terminal's PTY.
        for tab in &self.tabs {
            let _ = tab.notifier.0.send(Msg::Shutdown);
        }
    }
}

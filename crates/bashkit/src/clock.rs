//! Pausable execution clock for tracking active script execution time.
//!
//! The [`ExecutionClock`] tracks wall-clock time minus any paused intervals.
//! This allows external async operations (permission callbacks, user prompts)
//! to pause the clock so their wait time doesn't count against the execution
//! timeout budget.
//!
//! # Design
//!
//! - **Zero overhead when not paused** — `elapsed()` is a simple subtraction
//! - **RAII-safe** — [`PauseGuard`] resumes the clock on drop, even on panic
//! - **Nestable** — multiple pauses can be active simultaneously (e.g., a
//!   pipeline with parallel network requests each awaiting permission)
//! - **Thread-safe** — uses `std::sync::Mutex` (never held across awaits)

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// A wall-clock tracker that can be paused during external waits.
///
/// Tracks "active execution time" = wall time − total paused time.
/// Pausing is cooperative: callers acquire a [`PauseGuard`] that resumes
/// the clock when dropped.
///
/// # Example
///
/// ```rust
/// use bashkit::ExecutionClock;
/// use std::time::Duration;
///
/// let clock = ExecutionClock::new();
///
/// // Clock is running — elapsed time increases
/// assert!(!clock.is_expired(Duration::from_secs(10)));
///
/// // Pause the clock while waiting for external input
/// let guard = clock.pause();
/// // ... time passes but doesn't count ...
/// drop(guard);
///
/// // Clock resumes
/// ```
#[derive(Clone)]
pub struct ExecutionClock {
    inner: Arc<Mutex<ClockState>>,
}

struct ClockState {
    start: Instant,
    /// Accumulated paused duration from completed pauses.
    total_paused: Duration,
    /// Number of active pause guards.
    active_pauses: u32,
    /// When the current pause interval started (set when active_pauses goes 0 → 1).
    pause_started_at: Option<Instant>,
}

/// RAII guard that keeps the execution clock paused.
///
/// The clock resumes when this guard is dropped. Guards are nestable:
/// the clock only resumes when the *last* guard is dropped.
pub struct PauseGuard {
    clock: ExecutionClock,
}

impl ExecutionClock {
    /// Create a new clock that starts running immediately.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(ClockState {
                start: Instant::now(),
                total_paused: Duration::ZERO,
                active_pauses: 0,
                pause_started_at: None,
            })),
        }
    }

    /// Elapsed active execution time (wall time minus paused time).
    ///
    /// If the clock is currently paused, the ongoing pause interval is
    /// excluded from the returned duration.
    pub fn elapsed(&self) -> Duration {
        let state = self.inner.lock().expect("clock mutex poisoned");
        let wall = state.start.elapsed();
        let mut paused = state.total_paused;
        // If currently paused, also subtract the ongoing pause interval
        if let Some(pause_start) = state.pause_started_at {
            paused += pause_start.elapsed();
        }
        wall.saturating_sub(paused)
    }

    /// Check if active execution time has exceeded the given timeout.
    pub fn is_expired(&self, timeout: Duration) -> bool {
        self.elapsed() >= timeout
    }

    /// Pause the clock. Returns a guard that resumes it on drop.
    ///
    /// Multiple pauses can be active simultaneously (nested). The clock
    /// only resumes when the last guard is dropped.
    pub fn pause(&self) -> PauseGuard {
        let mut state = self.inner.lock().expect("clock mutex poisoned");
        state.active_pauses += 1;
        if state.active_pauses == 1 {
            // First pause — record when it started
            state.pause_started_at = Some(Instant::now());
        }
        PauseGuard {
            clock: self.clone(),
        }
    }

    /// Resume from a pause (called by PauseGuard::drop).
    fn resume(&self) {
        let mut state = self.inner.lock().expect("clock mutex poisoned");
        debug_assert!(state.active_pauses > 0, "resume without matching pause");
        state.active_pauses = state.active_pauses.saturating_sub(1);
        if state.active_pauses == 0 {
            // Last pause guard dropped — accumulate the paused duration
            if let Some(pause_start) = state.pause_started_at.take() {
                state.total_paused += pause_start.elapsed();
            }
        }
    }
}

impl Default for ExecutionClock {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for ExecutionClock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self.inner.lock().expect("clock mutex poisoned");
        f.debug_struct("ExecutionClock")
            .field("elapsed", &self.elapsed())
            .field("total_paused", &state.total_paused)
            .field("active_pauses", &state.active_pauses)
            .finish()
    }
}

impl Drop for PauseGuard {
    fn drop(&mut self) {
        self.clock.resume();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clock_starts_running() {
        let clock = ExecutionClock::new();
        // Elapsed should be very small but non-negative
        assert!(clock.elapsed() < Duration::from_secs(1));
        assert!(!clock.is_expired(Duration::from_secs(10)));
    }

    #[test]
    fn test_pause_stops_accumulation() {
        let clock = ExecutionClock::new();
        let guard = clock.pause();

        // While paused, sleep a bit — this should NOT count as execution time
        std::thread::sleep(Duration::from_millis(50));
        let elapsed_while_paused = clock.elapsed();

        drop(guard);

        // Active time should be very small (< 10ms) despite 50ms wall time
        assert!(
            elapsed_while_paused < Duration::from_millis(10),
            "elapsed while paused was {:?}, expected < 10ms",
            elapsed_while_paused
        );
    }

    #[test]
    fn test_resume_resumes_accumulation() {
        let clock = ExecutionClock::new();

        // Pause and resume
        let guard = clock.pause();
        std::thread::sleep(Duration::from_millis(20));
        drop(guard);

        // Now sleep without pausing — this SHOULD count
        std::thread::sleep(Duration::from_millis(50));
        let elapsed = clock.elapsed();

        // Should be >= 50ms (the unpaused sleep) but < 80ms (excluding paused time)
        assert!(
            elapsed >= Duration::from_millis(40),
            "elapsed {:?} should be >= 40ms",
            elapsed
        );
    }

    #[test]
    fn test_nested_pauses() {
        let clock = ExecutionClock::new();

        let guard1 = clock.pause();
        std::thread::sleep(Duration::from_millis(10));

        let guard2 = clock.pause();
        std::thread::sleep(Duration::from_millis(10));

        // Drop inner guard — clock should still be paused (guard1 still held)
        drop(guard2);
        std::thread::sleep(Duration::from_millis(10));
        let still_paused = clock.elapsed();

        // Drop outer guard — clock resumes
        drop(guard1);

        // All 30ms of sleep should have been paused
        assert!(
            still_paused < Duration::from_millis(10),
            "elapsed {:?} should be < 10ms (still paused)",
            still_paused
        );
    }

    #[test]
    fn test_is_expired() {
        let clock = ExecutionClock::new();
        assert!(!clock.is_expired(Duration::from_secs(10)));

        // With a very short timeout, it should expire quickly
        std::thread::sleep(Duration::from_millis(10));
        assert!(clock.is_expired(Duration::from_millis(5)));
    }

    #[test]
    fn test_is_expired_paused_doesnt_expire() {
        let clock = ExecutionClock::new();
        let guard = clock.pause();

        // Even after sleeping, the clock shouldn't expire because it's paused
        std::thread::sleep(Duration::from_millis(50));
        assert!(
            !clock.is_expired(Duration::from_millis(20)),
            "paused clock should not expire"
        );

        drop(guard);
    }

    #[test]
    fn test_clone_shares_state() {
        let clock1 = ExecutionClock::new();
        let clock2 = clock1.clone();

        let guard = clock1.pause();
        std::thread::sleep(Duration::from_millis(30));

        // Both clones see the paused state
        assert!(clock2.elapsed() < Duration::from_millis(10));

        drop(guard);
    }
}

use crate::environment::{Action, GameState};
use gym_rs::ActionType;

pub enum Stage {
    Train,
    Test,
}

/// Describes an Agent that knows how to interact with a Gym Environment
pub trait Agent {
    /// Given the current state, outputs the action the agent would like to take
    fn choose_action(&mut self, state: GameState) -> Action;
}


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

/// An Alternating Agent who always chooses a different action mostly used to make sure I know how to use
/// the gym API.
#[derive(Default)]
struct AlternatingAgent {
    alternate: bool,
}

impl Agent for AlternatingAgent {
    fn choose_action(&mut self, _state: GameState) -> Action {
        let action = if self.alternate {
            Action(ActionType::Discrete(0))
        } else {
            Action(ActionType::Discrete(1))
        };
        self.alternate = !self.alternate;
        action
    }
}

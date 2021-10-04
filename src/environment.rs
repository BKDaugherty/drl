use gym_rs::ActionType;
use std::clone::Clone;

pub const ACTION_SPACE: u8 = 2;
pub const OBSERVATION_SPACE: i64 = 4;

pub struct Action(pub ActionType);

impl From<ActionType> for Action {
    fn from(value: ActionType) -> Action {
        Action(value)
    }
}

impl Clone for Action {
    fn clone(&self) -> Self {
        match &self.0 {
            ActionType::Discrete(value) => Action(ActionType::Discrete(*value)),
            ActionType::Continuous(values) => Action(ActionType::Continuous(values.clone())),
        }
    }
}

/// The reward that the agent gets
#[derive(Clone, Debug)]
pub struct Reward(pub f64);

#[derive(Clone, Debug)]
pub enum GameStatus {
    Terminal,
    Ongoing,
}

// Gamestate could in theory be preprocessed if we wanted as they do in the paper
// Should this be a boool or enum?
#[derive(Clone, Debug)]
pub struct GameState {
    pub status: GameStatus,
    pub state: Vec<f64>,
}

impl GameState {
    pub fn new(state: Vec<f64>, game_complete: bool) -> Self {
        Self {
            state,
            status: match game_complete {
                true => GameStatus::Terminal,
                false => GameStatus::Ongoing,
            },
        }
    }
}

/// A Structure to capture all of the stuff going on in an episode
pub struct EpisodeResult {
    /// The total reward achieved by the agent in this episode
    pub total_reward: f64,
}

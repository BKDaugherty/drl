use crate::agent::Agent;
use anyhow::{Context, Result};
use gym_rs::{ActionType, GifRender, GymEnv};
use log::info;
use std::clone::Clone;

#[derive(Debug, Default, Clone)]
/// A Structure to capture all of the stuff going on in an episode
pub struct EpisodeResult {
    /// The total reward achieved by the agent in this episode
    pub total_reward: f64,
    pub learning_summaries: Vec<LearningResult>,
}

#[derive(Debug, Clone, Default)]
pub struct LearningResult {
    /// The loss
    pub mean_loss: f64,
}

// Required because ActionType doesn't impl Clone.
// Pull Request: https://github.com/MathisWellmann/gym-rs/pull/1
/// An Action taken in an environment
pub struct Action(pub ActionType);

impl From<ActionType> for Action {
    fn from(value: ActionType) -> Action {
        Action(value)
    }
}

/// The reward that the agent gets
#[derive(Clone, Debug)]
pub struct Reward(pub f64);

/// Whether the game is terminal or not
#[derive(Clone, Debug)]
pub enum GameStatus {
    Terminal,
    Ongoing,
}

/// The state of the game at any given time
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

impl Clone for Action {
    fn clone(&self) -> Self {
        match &self.0 {
            ActionType::Discrete(value) => Action(ActionType::Discrete(*value)),
            ActionType::Continuous(values) => Action(ActionType::Continuous(values.clone())),
        }
    }
}

pub fn run_episode(
    agent: &mut dyn Agent,
    env: &mut dyn GymEnv,
    render: &mut Option<GifRender>,
) -> Result<EpisodeResult> {
    let mut total_reward = 0.0;
    // Reset the environment before running through an episode
    let mut game_state = GameState::new(env.reset(), false);

    // Run the agent on the environment
    loop {
        let action = agent.choose_action(game_state);

        // Run a step of the environment
        let (next_state, reward, game_complete, info) = env.step(action.0);

        total_reward += reward;

        if let Some(info) = info {
            info!("Step information: {}", info);
        }
        // Advance the game state
        let next_state = GameState::new(next_state, game_complete);
        game_state = next_state;

        // Render the frame
        if let Some(render) = render.as_mut() {
            env.render(render);
        }

        // Break if the game is complete
        if game_complete {
            break;
        }
    }

    Ok(EpisodeResult {
        total_reward,
        learning_summaries: Vec::new(),
    })
}

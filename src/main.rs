use anyhow::{Context, Result};
use env_logger::Env;
use gym_rs::{ActionType, CartPoleEnv, GifRender, GymEnv};
use log::info;
use std::time::SystemTime;

type GameState = Vec<f64>;

/// Describes an Agent that knows how to interact with a Gym Environment
trait Agent {
    /// Given the current state, outputs the action the agent would like to take
    fn choose_action(&mut self, state: GameState) -> ActionType;
}

/// A Fixed Agent who always chooses the same action mostly used to make sure I know how to use
/// the gym API.
struct FixedAgent {}

impl Agent for FixedAgent {
    fn choose_action(&mut self, _state: GameState) -> ActionType {
        ActionType::Discrete(0)
    }
}

#[derive(Default)]
struct AlternatingAgent {
    alternate: bool,
}

impl Agent for AlternatingAgent {
    fn choose_action(&mut self, _state: GameState) -> ActionType {
        let action = if self.alternate {
            ActionType::Discrete(0)
        } else {
            ActionType::Discrete(1)
        };
        self.alternate = !self.alternate;
        action
    }
}

/// A Structure to capture all of the stuff going on in an episode
struct EpisodeResult {
    /// The total reward achieved by the agent in this episode
    total_reward: f64,
}

fn run_episode(
    agent: &mut dyn Agent,
    env: &mut dyn GymEnv,
    render: &mut Option<GifRender>,
) -> Result<EpisodeResult> {
    let mut total_reward = 0.0;
    // Reset the environment before running through an episode
    let mut game_state = env.reset();

    // Run the agent on the environment
    loop {
        let action = agent.choose_action(game_state);

        // Run a step of the environment
        let (next_state, reward, game_complete, info) = env.step(action);

        total_reward += reward;

        if let Some(info) = info {
            info!("Step information: {}", info);
        }

        // Advance the game state
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

    Ok(EpisodeResult { total_reward })
}

fn main() -> Result<()> {
    // Create our logger
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();

    // Create the cart pole environment
    let mut env = CartPoleEnv::default();

    // Get a timestamp to mark our results
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .expect("time went backwards")
        .as_secs();

    let render = GifRender::new(540, 540, &format!("img/cart_pole_{}.gif", now), 20).unwrap();

    // Create a dummy agent to test how this is all working
    let mut agent = AlternatingAgent::default();

    let result = run_episode(&mut agent, &mut env, &mut Some(render)).context("Running episode")?;

    info!(
        "Model finished the game with a total reward of {}",
        result.total_reward
    );

    Ok(())
}

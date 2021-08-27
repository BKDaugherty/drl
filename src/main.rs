extern crate tensorflow;

use anyhow::{Context, Result};
use env_logger::Env;
use gym_rs::{ActionType, CartPoleEnv, GifRender, GymEnv};
use log::{error, info, LevelFilter};
use std::time::SystemTime;
use tch::nn::Module;
use tch::IndexOp;
use tch::{nn, Tensor};

const ACTION_SPACE: u8 = 2;
const OBSERVATION_SPACE: i64 = 4;

/// The reward that the agent gets
struct Reward(f64);

// Gamestate could in theory be preprocessed if we wanted as they do in the paper
type GameState = Vec<f64>;

// TODO: Bikeshed
struct SequenceNode {
    state: GameState,
    action: ActionType,
}

///
struct Sequence {
    nodes: Vec<SequenceNode>,
}

/// Describes an Agent that knows how to interact with a Gym Environment
trait Agent {
    /// Given the current state, outputs the action the agent would like to take
    fn choose_action(&mut self, state: GameState) -> ActionType;
}

#[derive(Debug)]
/// A structure representing an action value function Q as defined in the paper.
struct ActionValueFunction {
    /// A neural network representing the parameters Q
    network: nn::Sequential,
}

/// The pytorch neural network implementation for the ActionValueFunction
impl Module for ActionValueFunction {
    fn forward(&self, start: &Tensor) -> Tensor {
        todo!()
    }
}

impl ActionValueFunction {
    fn new(var_store_path: &nn::Path) -> Self {
        let input_layer = nn::linear(
            var_store_path / "input_layer",
            OBSERVATION_SPACE,
            24,
            Default::default(),
        );
        let second_layer = nn::linear(var_store_path / "second_layer", 24, 24, Default::default());
        let output_layer = nn::linear(var_store_path / "output_layer", 24, 2, Default::default());

        let network = nn::seq()
            .add(input_layer)
            .add_fn(|xs| xs.relu())
            .add(second_layer)
            .add_fn(|xs| xs.relu())
            .add(output_layer);
        Self { network }
    }

    // Do I need to do some kind of no_grad thing here?
    fn choose_action(self, state: GameState) -> ActionType {
        // Convert the given game state into a tensor
        // I did this incorrectly... Could I somehow make applying matrices incorrectly a compile time error? Or at least a handled error?
        let observation = Tensor::of_slice(state.as_slice())
            .view_(&[1, OBSERVATION_SPACE])
            .to_kind(tch::Kind::Float);

        info!("Encoded observation {:?}", observation);
        // Send the tensor through a forward pass of the network
        let output = self.network.forward(&observation);
        info!("Result of forward pass {:?}", output);
        let choices = output.squeeze();
        let arg_max = choices.argmax(0, false);
	match u8::from(arg_max) {
	    a if a >= 0 || a <= ACTION_SPACE => ActionType::Discrete(a),
	    unknown =>  {
		error!("Unknown action type {}", unknown);
		// TODO -> Make this Result
		panic!("Unknown action type {}", unknown);
	    }
	}
    }
}

/// A structure implementing the DQN algorithm using experience replay
struct DQNAgent {
    /// Replay memory of sequence and action taken to result
    replay_memory: Vec<(Sequence, ActionType, Reward, GameState)>,
    /// The model the agent will use
    model: ActionValueFunction,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn init() {
        let _ = env_logger::builder()
            .filter(None, LevelFilter::Info)
            .is_test(true)
            .try_init();
    }

    #[test]
    fn test_network_shapes() {
        init();
        let vs = nn::VarStore::new(tch::Device::Cpu);
        let network = ActionValueFunction::new(&vs.root());

        let initial_game_state = vec![1.0, 1.0, 1.0, 1.0];
        let action = network.choose_action(initial_game_state);
        match action {
            ActionType::Discrete(action) => {
		info!("Correctly chose a discrete action!");
            }
            _ => panic!("Unexpected action type"),
        }
    }
}

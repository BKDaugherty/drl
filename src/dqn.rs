use anyhow::Result;
use gym_rs::ActionType;
use log::{error, info};
use tch::nn::Module;
use tch::{nn, Tensor};

use crate::agent::Stage;
use crate::environment::{
    Action, EpisodeResult, GameState, GameStatus, Reward, ACTION_SPACE, OBSERVATION_SPACE,
};

// TODO: Bikeshed
#[derive(Clone)]
pub struct SequenceNode {
    pub state: GameState,
    pub action: Action,
}

#[derive(Clone)]
pub struct Sequence {
    pub initial_state: GameState,
    pub nodes: Vec<SequenceNode>,
}

#[derive(Debug)]
/// A structure representing an action value function Q as defined in the paper.
pub struct ActionValueFunction {
    /// A neural network representing the parameters Q
    network: nn::Sequential,
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

    // Do I need to do some kind of no_grad thing here based on stage?
    pub fn choose_action(&self, state: GameState, _stage: Stage) -> Action {
        // Convert the given game state into a tensor
        // I did this incorrectly... Could I somehow make applying matrices incorrectly a compile time error? Or at least a handled error?
        let observation = Tensor::of_slice(state.state.as_slice())
            .view_(&[1, OBSERVATION_SPACE])
            .to_kind(tch::Kind::Float);

        info!("Encoded observation {:?}", observation);
        // Send the tensor through a forward pass of the network
        let output = self.network.forward(&observation);
        info!("Result of forward pass {:?}", output);
        let choices = output.squeeze();
        let arg_max = choices.argmax(0, false);
        match u8::from(arg_max) {
            a if a <= ACTION_SPACE => ActionType::Discrete(a).into(),
            unknown => {
                error!("Unknown action type {}", unknown);
                // TODO -> Make this Result
                panic!("Unknown action type {}", unknown);
            }
        }
    }
}

type MemoryNode = (Sequence, Action, Reward, GameState);

pub struct ReplayMemory {
    storage: Vec<MemoryNode>,
}

impl ReplayMemory {
    pub fn store(
        &mut self,
        previous_history: Sequence,
        action_taken: Action,
        reward: Reward,
        next_state: GameState,
    ) -> Result<()> {
        todo!()
    }
    pub fn sample(&self) -> Result<Vec<MemoryNode>> {
        todo!()
    }

    pub fn new() -> ReplayMemory {
        ReplayMemory { storage: vec![] }
    }
}

/// A structure implementing the DQN algorithm using experience replay
pub struct DQNAgent {
    /// Replay memory of sequence and action taken to result
    pub replay_memory: ReplayMemory,
    /// The model the agent will use
    pub model: ActionValueFunction,
}

impl DQNAgent {
    pub fn new(var_store_path: &nn::Path) -> DQNAgent {
        Self {
            replay_memory: ReplayMemory::new(),
            model: ActionValueFunction::new(var_store_path),
        }
    }
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
        match action.0 {
            ActionType::Discrete(action) => {
                info!("Correctly chose a discrete action!");
            }
            _ => panic!("Unexpected action type"),
        }
    }
}

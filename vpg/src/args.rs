use std::path::PathBuf;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
pub struct PerformArgs {
    /// The path from which to load the neural network
    #[structopt(long)]
    pub load_path: PathBuf,
}

#[derive(Debug, StructOpt)]
pub struct TrainArgs<H: StructOpt> {
    /// The path to which to save the trained neural network
    #[structopt(long, default_value = "./networks")]
    pub save_path: PathBuf,
    #[structopt(long, default_value = "5000")]
    pub episodes: u32,
    /// Number of training epochs to run
    #[structopt(long, default_value = "50")]
    pub epochs: u32,
    #[structopt(flatten)]
    pub hyperparameters: H,
    /// Whether or not we should plot results
    #[structopt(long)]
    pub plot_results: bool,
    /// Whether or not networks should be saved for later use
    #[structopt(long)]
    pub save_networks: bool,
}

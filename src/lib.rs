use ndarray::*;
use std::{
    collections::{BTreeMap, BTreeSet},
    iter::FromIterator,
};

pub type Player = usize;
pub type Strategy = usize;
pub type GameIndex = ArrayD<usize>;

#[derive(Debug, Clone)]
pub struct Game {
    grid: ArrayD<f32>,
}

impl Game {
    pub fn new(payoff_grid: ArrayD<f32>) -> Option<Game> {
        if Game::valid_shape(&payoff_grid) {
            Some(
                Game {
                    grid: payoff_grid,
                }
            )
        } else {
            None
        }
    }

    /// Checks if the grid has a valid shape for a game.
    /// The last dimention must have length of `ndim - 1`
    fn valid_shape<A>(grid: &ArrayD<A>) -> bool {
        let shape = grid.shape();
        let l = shape.len() - 1;
        shape[l] == l
    }

    pub fn is_valid(&self) -> bool {
        Game::valid_shape(&self.grid)
    }

    pub fn players(&self) -> impl Iterator<Item=Player> {
        0..(self.grid.shape().len() - 1)
    }

    pub fn strategies(&self, player: Player) -> impl Iterator<Item=Strategy> {
        0..self.num_strats(player)
    }

    fn num_strats(&self, player: Player) -> usize {
        self.grid.shape()[player]
    }

    /// Determines if s1 strictly dominates s2 for the given player
    /// This means that for all other players choices, s1 gives a better payoff than s2
    pub fn strictly_dominates(&self, player: Player, s1: Strategy, s2: Strategy) -> bool {
        self.compare_strats(|v1, v2| v1 > v2, player, s1, s2)
    }

    pub fn weakly_dominates(&self, player: Player, s1: Strategy, s2: Strategy) -> bool {
        self.compare_strats(|v1, v2| v1 >= v2, player, s1, s2)
    }


    fn compare_strats<F: Fn(f32, f32) -> bool>(&self, comparison: F, player: Player, s1: Strategy, s2: Strategy) -> bool {
        let payoff_axis = Axis(self.grid.shape().len() - 1);
        let player_payoffs = self
            .grid
            .index_axis(payoff_axis, player);
        let p1 = player_payoffs
            .index_axis(Axis(player), s1);
        let p2 = player_payoffs
            .index_axis(Axis(player), s2);
        let mut comp = ArrayD::from_elem(p1.shape(), false);
        Zip::from(&mut comp)
            .and(p1)
            .and(p2)
            .apply(|c, v1, v2| {
                *c = comparison(*v1, *v2);
            });
        comp.iter().all(|i| *i)
    }

    /// Constructs a new game by finding all the strictly dominated strategies, removing them, then
    /// repeating until there are no more.
    pub fn iterative_removal(&self) -> (Game, GameIndex) {
        let mut index_shape = Vec::from(self.grid.shape());
        index_shape.pop();
        let mut index = index_array(&index_shape);
        let mut mut_game = self.clone();

        loop {
            let dom_strats = mut_game.all_dominated_strats();
            let (updated_game, updated_index, any_updates) = mut_game.remove_strats(index, dom_strats);
            mut_game = updated_game;
            index = updated_index;

            if !any_updates {
                break;
            }
        }
        (mut_game, index)
    }

    /// Removes strategies from both the game and the game index.
    /// **Returns** (Game, Index, any_changes) where
    /// **Game** is the updated game
    /// **Index** is the index that tells you which strategies in the original game each choice
    /// corresponds to
    /// **any_changes** is true when there was at least 1 strategy removed
    fn remove_strats(mut self, mut index: GameIndex, by_player: BTreeMap<Player, BTreeSet<Strategy>>) -> (Self, GameIndex, bool) {
        let mut any_changes = false;
        for (player, to_remove) in by_player {
            any_changes |= to_remove.len() > 0;
            let all_strats = BTreeSet::from_iter(self.strategies(player));
            let to_keep: Vec<usize> = all_strats.difference(&to_remove).map(|x| *x).collect();
            self.grid = self.grid.select(Axis(player), to_keep.as_slice());
            index = index.select(Axis(player), to_keep.as_slice());
        }

        (self, index, any_changes)
    }

    /// Enumerates all the strategies that are strictly dominated by at least one other strategy.
    /// If a player is in the game, but does not have a value in the map, it means that the player
    /// has not dominated strategies.
    fn all_dominated_strats(&self) -> BTreeMap<Player, BTreeSet<Strategy>> {
        let mut by_player = BTreeMap::new();
        for player in self.players() {
            let mut dominated_strats = BTreeSet::new();
            for strat in self.strategies(player) {
                let mut alts = (0..self.num_strats(player))
                    .filter(|s| *s != strat);
                let dominates = |alt_strat| {
                    self.strictly_dominates(player, alt_strat, strat)
                };
                if alts.any(dominates) {
                    dominated_strats.insert(strat);
                    break;
                }
            }

            // if there are no dominated strats, we can leave this player empty
            if dominated_strats.len() > 0 {
                by_player.insert(player, dominated_strats);
            }
        }

        by_player
    }
}

/// Constructs an index array of the given shape with and added dimention for the index values.
/// The last dimention is of length `shape.len()`.
/// The values describe the coordinate of the index.
fn index_array<'a>(shape: &'a [usize]) -> GameIndex {
    let mut shape = Vec::from(shape);
    let num_dims = shape.len();
    shape.push(num_dims);
    ArrayD::from_shape_fn(shape, |i| {
        let index_num = i.ndim() - 1;
        let cdim = i[index_num];
        i[cdim]
    })
}

#[cfg(test)]
mod test {
    use super::*;

    fn prisoners_dilemma() -> Game {
        let pd_grid = ArrayD::from_shape_fn(vec![2,2,2], |dim| {
            10.0 - match (dim[0], dim[1], dim[2]) {
                (0, 0, _) => 5.0,
                (1, 0, 0) => 7.0,
                (1, 0, 1) => 0.0,
                (0, 1, 0) => 0.0,
                (0, 1, 1) => 7.0,
                (1, 1, _) => 2.0,
                _ => 100.0,
            }
        });

        Game::new(pd_grid).unwrap()
    }

    #[test]
    fn test_strictly_dominates() {
        let pd = prisoners_dilemma();
        assert!(pd.strictly_dominates(0,0,1), "Action 0 (telling) should strictly dominate action 1");
        assert!(!pd.strictly_dominates(0,1,0), "Action 1 (keeping quiet) should not strictly dominate action 0");
    }

    #[test]
    fn test_num_players() {
        let pd = prisoners_dilemma();
        assert_eq!(pd.players().count(), 2);
    }

    #[test]
    fn test_num_actions() {
        let pd = prisoners_dilemma();
        assert_eq!(pd.strategies(0).count(), 2);
        assert_eq!(pd.strategies(1).count(), 2);
    }

    #[test]
    fn test_all_dominated_starts() {
        let pd = prisoners_dilemma();
        let dominated_strats = pd.all_dominated_strats();
        assert_eq!(dominated_strats.iter().count(), 2, "2 unique players should have dominated starts");
        assert!(dominated_strats[&0].contains(&1), "player 0, strat 1 is dominated by strat 0");
    }

    #[test]
    fn test_iterative_removal() {
        let pd = prisoners_dilemma();
        let (reduced, lookup) = pd.iterative_removal();

        assert!(reduced.is_valid());

        // there should only be one remaining strat for each player
        assert_eq!(reduced.strategies(0).count(), 1);
        assert_eq!(reduced.strategies(1).count(), 1);

        // the strategy should be they both tell
        assert_eq!(lookup[Dim((0,0,0))], 0);
        assert_eq!(lookup[Dim((0,0,1))], 0);
    }

    #[test]
    fn test_index_array() {
        const ROWS: usize = 10;
        const COLS: usize = 10;
        let shape = vec![ROWS, COLS];
        let index = index_array(&shape);

        assert_eq!(index.shape(), &[ROWS,COLS,2], "index shape should match (2,2,2)");
        for row in 0..ROWS {
            for col in 0..COLS {
                assert_eq!(index[Dim((row, col, 0))], row);
                assert_eq!(index[Dim((row, col, 1))], col);
            }
        }
    }
}

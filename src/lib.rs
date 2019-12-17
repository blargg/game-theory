use float_cmp::*;
use ndarray::*;
use std::{
    collections::{BTreeMap, BTreeSet},
    iter::FromIterator,
};

pub type Player = usize;
pub type Strategy = usize;
pub type GameIndex = ArrayD<usize>;

/// Defines the utility value for all players.
#[derive(Debug, Clone)]
pub struct Util {
    values: Vec<f32>,
}

impl Util {
    /// Constructs a new util, using the given values. The values are 0 indexed (first value is
    /// player 0).
    pub fn from_vec(values: Vec<f32>) -> Util {
        Util {
            values,
        }
    }

    /// Gets the Util for a given player. If out of bounds, the util is 0.
    pub fn for_player(&self, player: Player) -> f32 {
        if player < self.values.len() {
            self.values[player]
        } else {
            0.0
        }
    }
}

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

    pub fn from_util_array(util_array: ArrayD<Util>) -> Game {
        let util_axis = util_array.shape().len();
        let mut exp_shape: Vec<_> = util_array.shape().iter().cloned().collect();
        exp_shape.push(util_array.shape().len());
        let expanded_array = ArrayD::from_shape_fn(exp_shape, |i| {
            let player = i[util_axis];
            let util_index = i.remove_axis(Axis(util_axis));
            util_array[util_index].for_player(player)
        });

        Game {
            grid: expanded_array,
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

    fn num_players(&self) -> usize {
        self.grid.shape().len() - 1
    }

    pub fn strategies(&self, player: Player) -> impl Iterator<Item=Strategy> {
        0..self.num_strats(player)
    }

    fn num_strats(&self, player: Player) -> usize {
        self.grid.shape()[player]
    }

    pub fn util(&self, player: Player, strat_index: &Vec<Strategy>) -> f32 {
        let mut player_util_index = strat_index.clone();
        player_util_index.push(player);

        let d = Dim(player_util_index);
        self.grid[d]
    }

    /// Returns the utilties for all players for the given strategic index
    pub fn utilities(&self, strat_index: &Vec<Strategy>) -> Util {
        let mut util = Vec::with_capacity(self.num_players());
        for player in self.players() {
            let player_util = self.util(player, &strat_index);
            util.push(player_util);
        }

        Util::from_vec(util)
    }

    /// Determines if s1 strictly dominates s2 for the given player
    /// This means that for all other players choices, s1 gives a better payoff than s2
    pub fn strictly_dominates(&self, player: Player, s1: Strategy, s2: Strategy) -> bool {
        let util_comp = self.compare_strats(player, s1, s2);
        util_comp.iter().all(|(v1, v2)| v1 > v2)
    }

    pub fn weakly_dominates(&self, player: Player, s1: Strategy, s2: Strategy) -> bool {
        let util_comp = self.compare_strats(player, s1, s2);
        util_comp.iter().all(|(v1, v2)| v1 >= v2) &&
            util_comp.iter().any(|(v1, v2)| v1 > v2)
    }

    fn dominates(&self, strictly: bool, player: Player, s1: Strategy, s2: Strategy) -> bool {
        if strictly {
            self.strictly_dominates(player, s1, s2)
        } else {
            self.weakly_dominates(player, s1, s2)
        }
    }

    /// Zips together two strategies into a view to compare utilities for other player strategies.
    /// **Returns** an array view of pairs of utilities from (s1, s2) respectively.
    fn compare_strats<'a>(&'a self, player: Player, s1: Strategy, s2: Strategy) -> ArrayD<(f32, f32)> {
        let payoff_axis = Axis(self.grid.shape().len() - 1);
        let player_payoffs = self
            .grid
            .index_axis(payoff_axis, player);
        let p1 = player_payoffs
            .index_axis(Axis(player), s1);
        let p2 = player_payoffs
            .index_axis(Axis(player), s2);
        let mut comp = ArrayD::from_elem(p1.shape(), (0.0, 0.0));
        Zip::from(&mut comp)
            .and(p1)
            .and(p2)
            .apply(|c, v1, v2| {
                *c = (*v1, *v2);
            });
        comp
    }

    pub fn strict_iterative_removal(&self) -> (Game, GameIndex) {
        self.iterative_removal(true)
    }

    pub fn weak_iterative_removal(&self) -> (Game, GameIndex) {
        self.iterative_removal(false)
    }

    /// Constructs a new game by finding all the strictly dominated strategies, removing them, then
    /// repeating until there are no more.
    pub fn iterative_removal(&self, strictly: bool) -> (Game, GameIndex) {
        let mut index_shape = Vec::from(self.grid.shape());
        index_shape.pop();
        let mut index = index_array(&index_shape);
        let mut mut_game = self.clone();

        loop {
            let dom_strats = mut_game.all_dominated_strats(strictly);
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
    fn all_dominated_strats(&self, strictly: bool) -> BTreeMap<Player, BTreeSet<Strategy>> {
        let mut by_player = BTreeMap::new();
        for player in self.players() {
            let mut dominated_strats = BTreeSet::new();
            for strat in self.strategies(player) {
                let mut alts = (0..self.num_strats(player))
                    .filter(|s| *s != strat);
                let dominates = |alt_strat| {
                    self.dominates(strictly, player, alt_strat, strat)
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

    /// Determines if two strategies are identical, meaning for all players, the given players
    /// choice of s1 or s2 is effectivly the same.
    pub fn identical_utils(&self, player: Player, s1: Strategy, s2: Strategy) -> bool {
        let u1 = self
            .grid
            .index_axis(Axis(player), s1);
        let u2 = self
            .grid
            .index_axis(Axis(player), s2);

        let mut all_eq = true;
        Zip::from(u1)
            .and(u2)
            .apply(|u1, u2| {
                all_eq = all_eq && approx_eq!(f32, *u1, *u2);
            });

        all_eq
    }

    /// Finds all the duplicate strategies. That is to say, all strategies that have some identical
    /// strategy in the game, except for on representative version of that strat.
    fn all_duplicates(&self) -> BTreeMap<Player, BTreeSet<Strategy>> {
        let mut by_player = BTreeMap::new();
        for player in self.players() {
            let mut duplicate_strats = BTreeSet::new();
            for strat in self.strategies(player) {
                for alt in strat+1..self.num_strats(player) {
                    if self.identical_utils(player, strat, alt) {
                        duplicate_strats.insert(alt);
                    }
                }
            }

            if duplicate_strats.len() > 0 {
                by_player.insert(player, duplicate_strats);
            }
        }

        by_player
    }

    pub fn remove_duplicate_strats(self, index: GameIndex) -> (Game, GameIndex) {
        let dups = self.all_duplicates();
        let (new_game, new_index, _any_removed) = self.remove_strats(index, dups);
        (new_game, new_index)
    }

    /// Constructs an index for the current shape of the game.
    /// See `index_array` for more information.
    pub fn index(&self) -> GameIndex {
        let shape = self.grid.shape();
        index_array(shape)
    }

    /// Checks if the given strategic profile is a nash equilibrium.
    pub fn is_nash_eq(&self, prof: &Vec<Strategy>) -> bool {
        assert_eq!(prof.len(), self.num_players(), "profile must have a strategy selected for every player");

        let strict_highest = |player| {
            self.stict_highest_util(player, prof[player], prof)
        };

        self.players().all(strict_highest)
    }

    /// Checks if for the given player and strategic profile, that the selected strategy **strat**
    /// is the highest utility strategy, assuming only **player** can change their strategy
    fn stict_highest_util(&self, player: Player, strat: Strategy, prof: &Vec<Strategy>) -> bool {
        let payoff_axis = Axis(self.grid.shape().len() - 1);
        let player_payoffs = self
            .grid
            .index_axis(payoff_axis, player);
        let selected = Dim(prof.clone());
        let selected_val = player_payoffs[selected];
        let mut cur_index = prof.clone();
        for alt_strat in self.strategies(player).filter(|st| *st != strat) {
            cur_index[player] = alt_strat;
            let cur_value = player_payoffs[Dim(cur_index.clone())];

            if cur_value >= selected_val {
                return false;
            }
        }

        true
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

    /// Constructs a version of the prisoners dilemma
    /// **Players** 2
    /// **Actions** all players can either tell (action 0) or withhold info (action 1)
    /// **Utilities** negative number of years that player spends in jail (negative, since less is
    /// better)
    fn prisoners_dilemma() -> Game {
        let pd_grid = ArrayD::from_shape_fn(vec![2,2,2], |dim| {
            -1.0 * match (dim[0], dim[1], dim[2]) {
                (0, 0, _) => 5.0,
                (1, 0, 0) => 7.0,
                (1, 0, 1) => 0.0,
                (0, 1, 0) => 0.0,
                (0, 1, 1) => 7.0,
                (1, 1, _) => 2.0,
                _ => panic!("Unexpected index while constructing prisoner's dilemma"),
            }
        });

        Game::new(pd_grid).unwrap()
    }

    /// Constructs an example game where multiple rounds of elimination must take place.
    fn multiple_weak_elim() -> Game {
        let grid = ArrayD::from_shape_fn(vec![3, 3, 2], |dim| {
            match (dim[0], dim[1], dim[2]) {
                (0, 0, _) => 1.0,
                (0, 1, 0) => 0.0,
                (0, 1, 1) => 1.0,
                (0, 2, 0) => 3.0,
                (0, 2, 1) => 1.0,
                (1, 0, 0) => 1.0,
                (1, 0, 1) => 0.0,
                (1, 1, _) => 2.0,
                (1, 2, 0) => 1.0,
                (1, 2, 1) => 3.0,
                (2, 0, 0) => 1.0,
                (2, 0, 1) => 3.0,
                (2, 1, 0) => 3.0,
                (2, 1, 1) => 1.0,
                (2, 2, _) => 2.0,
                _ => panic!("Unhandled index when constructing multiple weak elim game"),
            }
        });

        Game::new(grid).unwrap()
    }

    fn duplicate_strats() -> Game {
        let grid = ArrayD::from_shape_fn(vec![3,1,2], |dim| {
            match (dim[0], dim[1], dim[2]) {
                (0, 0, _) => 0.0,
                (1, 0, _) => 1.0,
                (2, 0, _) => 1.0,
                _ => panic!("Unhandled index when constructing duplicate strats game"),
            }
        });

        Game::new(grid).unwrap()
    }

    #[test]
    fn from_util_array_test() {
        let util_vec: Vec<Util> = vec![
            Util::from_vec(vec![0.0, 1.0]),
            Util::from_vec(vec![0.0, 0.0]),
            Util::from_vec(vec![0.0, 0.0]),
            Util::from_vec(vec![4.0, 4.1]),
        ];
        let utils = ArrayD::from_shape_vec(vec![2, 2], util_vec)
        .expect("from_shape_vec should work");

        let game = Game::from_util_array(utils);
        assert_eq!(0.0, game.grid[Dim((0,0,0))], "action 0,0, player 0 should be 0.0");

        let val = game.grid[Dim((0,0,1))];
        assert!(val > 0.9 && val < 1.1, "action 0,0, player 1 should be 1.0");

        let val = game.grid[Dim((1,1,0))];
        assert!(val > 3.9 && val < 4.05, "action 2,2, player 0 should be 4.0");

        let val = game.grid[Dim((1,1,1))];
        assert!(val > 4.05 && val < 4.15, "action 2,2, player 1 should be 4.1");
    }

    #[test]
    fn util_test() {
        let game = prisoners_dilemma();
        let both_tell_util = game.util(0, &vec![0,0]);
        assert!(approx_eq!(f32, -5.0, both_tell_util), "Both telling should have util of -5.0");
    }

    #[test]
    fn utils_test() {
        let game = prisoners_dilemma();
        let one_tell_util = game.utilities(&vec![0,1]);
        assert!(approx_eq!(f32, 0.0, one_tell_util.for_player(0)), "Telling player should have util of 0.0");
        assert!(approx_eq!(f32, -7.0, one_tell_util.for_player(1)), "Quiet player should have util of -7.0");
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
        let dominated_strats = pd.all_dominated_strats(true);
        assert_eq!(dominated_strats.iter().count(), 2, "2 unique players should have dominated starts");
        assert!(dominated_strats[&0].contains(&1), "player 0, strat 1 is dominated by strat 0");
    }

    #[test]
    fn test_iterative_removal() {
        let pd = prisoners_dilemma();
        let (reduced, lookup) = pd.strict_iterative_removal();

        assert!(reduced.is_valid());

        // there should only be one remaining strat for each player
        assert_eq!(reduced.strategies(0).count(), 1);
        assert_eq!(reduced.strategies(1).count(), 1);

        // the strategy should be they both tell
        assert_eq!(lookup[Dim((0,0,0))], 0);
        assert_eq!(lookup[Dim((0,0,1))], 0);
    }

    #[test]
    fn iterative_removal_multiple_test() {
        let game = multiple_weak_elim();
        let (reduced, lookup) = game.weak_iterative_removal();

        // the final game only has 1 strategy per player
        assert_eq!(reduced.strategies(0).count(), 1);
        assert_eq!(reduced.strategies(1).count(), 1);

        // the remaining strategies should be (0,0) in the original game
        assert_eq!(lookup[Dim((0,0,0))], 0, "player 0's remaining strat should be 0 in the original game");
        assert_eq!(lookup[Dim((0,0,1))], 0, "player 1's remaining strat should be 0 in the original game");
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

    #[test]
    fn remove_duplicate_strats_test() {
        let game = duplicate_strats();
        let index = game.index();

        let (reduced, _index) = game.remove_duplicate_strats(index);
        assert_eq!(reduced.strategies(0).count(), 2);
        assert_eq!(reduced.strategies(1).count(), 1);
    }

    #[test]
    fn is_nash_eq_test() {
        let game = prisoners_dilemma();
        assert!(game.is_nash_eq(&vec![0,0]), "both telling should be a nash eq");
        assert!(!game.is_nash_eq(&vec![1,1]), "both cooperating should not be a nash eq");
        assert!(!game.is_nash_eq(&vec![0,1]), "one telling, one coop should not be a nash eq");
    }
}

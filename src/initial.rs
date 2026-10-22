// SPDX-License-Identifier: Apache-2.0
//! Initial partitioning of the coarsest graph (`InitialPartition`): many random starts, each made
//! legal and improved by the k-way FM refiner, the cheapest kept.

use crate::evaluator::Evaluator;
use crate::hypergraph::{blocks_le, Hypergraph, Weights};
use crate::mt19937::Mt19937;
use crate::partitioner::{random_part, vile_part};
use crate::refiner::KWayFm;

/// How each solution was made, for the trace.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Random,
    RandomVile,
    Vile,
}

#[derive(Clone, Debug)]
pub struct InitialPartitions {
    /// Every solution tried, in the order made: kind, index within its kind, cut cost, and
    /// whether its blocks are all within the UPPER bounds.
    pub tried: Vec<(Kind, usize, f32, bool)>,
    /// The kept solutions, best first.
    pub top: Vec<Vec<usize>>,
    /// The cost of the best solution within bounds (0 when none is).
    pub best_cost: f32,
}

impl Kind {
    /// The reference's trace line for one tried solution.
    fn line(self, i: usize, cost: f32, flag: bool) -> String {
        match self {
            Kind::Random => format!("[initial_partitioning] {i} :: Random part cutcost = {cost}, balance_flag = {flag}"),
            Kind::RandomVile => format!("[initial_partitioning] {i} :: Random VILE part cutcost = {cost}, balance_flag = {flag}"),
            Kind::Vile => format!("[initial_partitioning] VILE part cutcost = {cost}, balance_flag = {flag}"),
        }
    }
}

/// Why a stage stopped short of an answer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refused {
    /// The graph is small enough that the reference also tries an exact (ILP) partition.
    NeedsIlp { num_vertices: usize },
}

/// The parameters `InitialPartition` reads.
pub struct InitialParams {
    pub num_parts: usize,
    /// The multilevel partitioner's seed — the same for every call.
    pub seed: i32,
    pub num_initial_random_solutions: usize,
    pub num_best_initial_solutions: usize,
    pub num_vertices_threshold_ilp: usize,
}

/// `InitialPartition`. One generator seeded once hands each random start its own seed: first the
/// balanced random starts, then the vile random ones, then one vile start; the refiner may make
/// as many moves as there are vertices until after the vile start (whose line comes AFTER the
/// refiner's reset lines). Kept: the cheapest solutions within bounds, ordered by (cost, order
/// made), topped up with the cheapest of the rest. Trace lines go to `trace` in the reference's
/// order. A graph small enough for the exact (ILP) start is refused before anything is printed
/// past the opening line.
pub fn initial_partition(
    p: &InitialParams,
    ev: &Evaluator,
    fm: &mut KWayFm,
    g: &Hypergraph,
    upper: &[Weights],
    lower: &[Weights],
    trace: &mut Vec<String>,
) -> Result<InitialPartitions, Refused> {
    trace.push("[initial_partitioning] Running Initial Partitioning...".into());
    if g.num_vertices() <= p.num_vertices_threshold_ilp {
        return Err(Refused::NeedsIlp { num_vertices: g.num_vertices() });
    }
    let mut gen = Mt19937::new(p.seed as u32);
    let mut solutions: Vec<Vec<usize>> = Vec::new();
    let mut tried = Vec::new();
    fm.set_max_move(g.num_vertices(), trace);
    let run = |solution: Vec<usize>, fm: &KWayFm, tried: &mut Vec<(Kind, usize, f32, bool)>, kind: Kind, i: usize, solutions: &mut Vec<Vec<usize>>| {
        let mut solution = solution;
        fm.refine(g, upper, lower, &mut solution);
        let (cost, balance) = ev.cut_evaluator(g, &solution);
        // Only the upper bound is checked, to keep more solutions in play.
        tried.push((kind, i, cost, blocks_le(&balance, upper)));
        solutions.push(solution);
    };
    for (kind, vile) in [(Kind::Random, false), (Kind::RandomVile, true)] {
        for i in 0..p.num_initial_random_solutions {
            let seed = (gen.next_u32() & i32::MAX as u32) as i32;
            run(random_part(g, p.num_parts, upper, lower, seed, vile), fm, &mut tried, kind, i, &mut solutions);
            let &(k, i, cost, flag) = tried.last().expect("just tried");
            trace.push(k.line(i, cost, flag));
        }
    }
    let mut vile_solution = vile_part(g, p.num_parts, ev);
    fm.refine(g, upper, lower, &mut vile_solution);
    fm.restore_default_parameters(trace);
    let (cost, balance) = ev.cut_evaluator(g, &vile_solution);
    tried.push((Kind::Vile, 0, cost, blocks_le(&balance, upper)));
    solutions.push(vile_solution);
    trace.push(Kind::Vile.line(0, cost, tried.last().expect("just tried").3));

    let mut ids: Vec<usize> = (0..tried.len()).collect();
    ids.sort_by(|&x, &y| tried[x].2.partial_cmp(&tried[y].2).expect("finite cost").then(x.cmp(&y)));
    let mut top = Vec::new();
    let mut chosen = vec![false; ids.len()];
    let mut best_cost = 0.0f32;
    for &id in &ids {
        if top.len() >= p.num_best_initial_solutions {
            break;
        }
        if tried[id].3 {
            if top.is_empty() {
                best_cost = tried[id].2;
            }
            top.push(solutions[id].clone());
            chosen[id] = true;
        }
    }
    for &id in &ids {
        if top.len() >= p.num_best_initial_solutions {
            break;
        }
        if !chosen[id] {
            top.push(solutions[id].clone());
            chosen[id] = true;
        }
    }
    trace.push(format!("[initial_partitioning] Number of chosen best initial solutions = {}", top.len()));
    trace.push(format!("[initial_partitioning] Best initial cutcost {best_cost}"));
    Ok(InitialPartitions { tried, top, best_cost })
}

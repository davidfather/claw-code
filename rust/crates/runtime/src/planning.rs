use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Display, Formatter};

use serde::{Deserialize, Serialize};

pub const DEFAULT_MAX_REPLANNING_ITERATIONS: u8 = 3;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskPlanning {
    pub objective: String,
    pub planned_tasks: Vec<PlannedTask>,
    pub acceptance_criteria: Vec<AcceptanceCriterion>,
    pub verification_evidence: Vec<VerificationEvidence>,
    pub completion_policy: CompletionPolicy,
    pub replanning: ReplanningState,
}

impl TaskPlanning {
    #[must_use]
    pub fn new(objective: impl Into<String>) -> Self {
        Self::with_max_replanning_iterations(objective, DEFAULT_MAX_REPLANNING_ITERATIONS)
    }

    #[must_use]
    pub fn with_max_replanning_iterations(
        objective: impl Into<String>,
        max_iterations: u8,
    ) -> Self {
        Self {
            objective: objective.into(),
            planned_tasks: Vec::new(),
            acceptance_criteria: Vec::new(),
            verification_evidence: Vec::new(),
            completion_policy: CompletionPolicy::default(),
            replanning: ReplanningState::new(max_iterations),
        }
    }

    #[must_use]
    pub fn required_tasks(&self) -> impl Iterator<Item = &PlannedTask> {
        self.planned_tasks.iter().filter(|task| task.required)
    }

    #[must_use]
    pub fn incomplete_required_tasks(&self) -> Vec<&PlannedTask> {
        self.required_tasks()
            .filter(|task| task.status != PlannedTaskStatus::Completed)
            .collect()
    }

    #[must_use]
    pub fn required_criteria(&self) -> impl Iterator<Item = &AcceptanceCriterion> {
        self.acceptance_criteria
            .iter()
            .chain(
                self.planned_tasks
                    .iter()
                    .flat_map(|task| task.acceptance_criteria.iter()),
            )
            .filter(|criterion| criterion.required)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlannedTask {
    pub id: String,
    pub title: String,
    pub status: PlannedTaskStatus,
    pub required: bool,
    pub depends_on: Vec<String>,
    pub expected_outputs: Vec<String>,
    pub acceptance_criteria: Vec<AcceptanceCriterion>,
    pub incomplete_reason: Option<IncompleteReason>,
}

impl PlannedTask {
    #[must_use]
    pub fn new(id: impl Into<String>, title: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            title: title.into(),
            status: PlannedTaskStatus::Pending,
            required: true,
            depends_on: Vec::new(),
            expected_outputs: Vec::new(),
            acceptance_criteria: Vec::new(),
            incomplete_reason: None,
        }
    }

    #[must_use]
    pub fn optional(mut self) -> Self {
        self.required = false;
        self
    }

    #[must_use]
    pub fn with_status(mut self, status: PlannedTaskStatus) -> Self {
        self.status = status;
        self
    }

    #[must_use]
    pub fn with_incomplete_reason(mut self, reason: IncompleteReason) -> Self {
        self.incomplete_reason = Some(reason);
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlannedTaskStatus {
    Pending,
    InProgress,
    Completed,
    Blocked,
    Deferred,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AcceptanceCriterion {
    pub id: String,
    pub description: String,
    pub required: bool,
}

impl AcceptanceCriterion {
    #[must_use]
    pub fn required(id: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            description: description.into(),
            required: true,
        }
    }

    #[must_use]
    pub fn optional(id: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            description: description.into(),
            required: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerificationEvidence {
    pub id: String,
    pub task_id: Option<String>,
    pub criterion_id: Option<String>,
    pub command: Option<String>,
    pub passed: bool,
    pub summary: String,
}

impl VerificationEvidence {
    #[must_use]
    pub fn passed(
        id: impl Into<String>,
        task_id: Option<String>,
        criterion_id: Option<String>,
        summary: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            task_id,
            criterion_id,
            command: None,
            passed: true,
            summary: summary.into(),
        }
    }

    #[must_use]
    pub fn failed(
        id: impl Into<String>,
        task_id: Option<String>,
        criterion_id: Option<String>,
        summary: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            task_id,
            criterion_id,
            command: None,
            passed: false,
            summary: summary.into(),
        }
    }

    #[must_use]
    pub fn with_command(mut self, command: impl Into<String>) -> Self {
        self.command = Some(command.into());
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IncompleteReason {
    TooBroad,
    MissingDependency,
    BlockedByError,
    MissingTool,
    VerificationFailed,
    UserInputRequired,
    OutOfScope,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompletionPolicy {
    pub require_all_required_tasks_complete: bool,
    pub require_required_criteria_evidence: bool,
    pub allow_partial_completion: bool,
}

impl Default for CompletionPolicy {
    fn default() -> Self {
        Self {
            require_all_required_tasks_complete: true,
            require_required_criteria_evidence: true,
            allow_partial_completion: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplanningState {
    pub iteration: u8,
    pub max_iterations: u8,
    pub history: Vec<ReplanningAttempt>,
}

impl ReplanningState {
    #[must_use]
    pub fn new(max_iterations: u8) -> Self {
        Self {
            iteration: 0,
            max_iterations,
            history: Vec::new(),
        }
    }

    #[must_use]
    pub fn can_replan(&self) -> bool {
        self.iteration < self.max_iterations
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplanningAttempt {
    pub iteration: u8,
    pub reason: String,
    pub changes: Vec<TaskPlanningChange>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TaskPlanningChange {
    AddTask {
        task: PlannedTask,
    },
    Split {
        from: String,
        into: Vec<PlannedTask>,
    },
    Replace {
        from: String,
        to: PlannedTask,
    },
    AddDependency {
        task_id: String,
        dependency_id: String,
    },
    MarkBlocked {
        task_id: String,
        reason: IncompleteReason,
    },
    MarkCompleted {
        task_id: String,
        evidence: Vec<VerificationEvidence>,
    },
    Defer {
        task_id: String,
        reason: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompletionDecision {
    pub status: CompletionDecisionStatus,
    pub incomplete_task_ids: Vec<String>,
    pub missing_criterion_ids: Vec<String>,
    pub failed_evidence_ids: Vec<String>,
    pub replanning_allowed: bool,
    pub reason: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompletionDecisionStatus {
    Complete,
    NeedsReplanning,
    NeedsUserInput,
    IncompleteMaxIterations,
    Incomplete,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplanningError {
    message: String,
}

impl ReplanningError {
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl Display for ReplanningError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for ReplanningError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanningCoordinator {
    planning: TaskPlanning,
}

impl PlanningCoordinator {
    #[must_use]
    pub fn new(planning: TaskPlanning) -> Self {
        Self { planning }
    }

    #[must_use]
    pub fn planning(&self) -> &TaskPlanning {
        &self.planning
    }

    #[must_use]
    pub fn into_planning(self) -> TaskPlanning {
        self.planning
    }

    #[must_use]
    pub fn evaluate_completion(&self) -> CompletionDecision {
        let incomplete_task_ids = if self
            .planning
            .completion_policy
            .require_all_required_tasks_complete
        {
            self.planning
                .incomplete_required_tasks()
                .into_iter()
                .map(|task| task.id.clone())
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        let failed_evidence_ids = self
            .planning
            .verification_evidence
            .iter()
            .filter(|evidence| !evidence.passed)
            .map(|evidence| evidence.id.clone())
            .collect::<Vec<_>>();
        let missing_criterion_ids = self.missing_required_criteria();
        let user_input_required = self.planning.required_tasks().any(|task| {
            task.status != PlannedTaskStatus::Completed
                && task.incomplete_reason == Some(IncompleteReason::UserInputRequired)
        });

        if incomplete_task_ids.is_empty()
            && missing_criterion_ids.is_empty()
            && failed_evidence_ids.is_empty()
        {
            return CompletionDecision {
                status: CompletionDecisionStatus::Complete,
                incomplete_task_ids,
                missing_criterion_ids,
                failed_evidence_ids,
                replanning_allowed: false,
                reason: "all required planned tasks and acceptance criteria are verified"
                    .to_string(),
            };
        }

        if user_input_required {
            return CompletionDecision {
                status: CompletionDecisionStatus::NeedsUserInput,
                incomplete_task_ids,
                missing_criterion_ids,
                failed_evidence_ids,
                replanning_allowed: false,
                reason: "at least one planned task requires user input".to_string(),
            };
        }

        let replanning_allowed = self.planning.replanning.can_replan();
        let status = if replanning_allowed {
            CompletionDecisionStatus::NeedsReplanning
        } else if self.planning.completion_policy.allow_partial_completion {
            CompletionDecisionStatus::Incomplete
        } else {
            CompletionDecisionStatus::IncompleteMaxIterations
        };
        let reason = if replanning_allowed {
            "planning is incomplete but another replanning iteration is available"
        } else {
            "planning is incomplete and maximum replanning iterations were reached"
        };

        CompletionDecision {
            status,
            incomplete_task_ids,
            missing_criterion_ids,
            failed_evidence_ids,
            replanning_allowed,
            reason: reason.to_string(),
        }
    }

    pub fn run_replanning_loop<F>(
        &mut self,
        mut replan: F,
    ) -> Result<CompletionDecision, ReplanningError>
    where
        F: FnMut(&TaskPlanning, &CompletionDecision) -> Result<ReplanningProposal, ReplanningError>,
    {
        loop {
            let decision = self.evaluate_completion();
            match decision.status {
                CompletionDecisionStatus::Complete
                | CompletionDecisionStatus::NeedsUserInput
                | CompletionDecisionStatus::Incomplete
                | CompletionDecisionStatus::IncompleteMaxIterations => return Ok(decision),
                CompletionDecisionStatus::NeedsReplanning => {
                    let proposal = replan(&self.planning, &decision)?;
                    if proposal.changes.is_empty() {
                        return Err(ReplanningError::new(
                            "replanning proposal must include at least one change",
                        ));
                    }
                    self.apply_replanning(proposal.reason, proposal.changes)?;
                }
            }
        }
    }

    pub fn apply_replanning(
        &mut self,
        reason: impl Into<String>,
        changes: Vec<TaskPlanningChange>,
    ) -> Result<ReplanningAttempt, ReplanningError> {
        if !self.planning.replanning.can_replan() {
            return Err(ReplanningError::new(format!(
                "maximum replanning iterations reached: {}",
                self.planning.replanning.max_iterations
            )));
        }
        if changes.is_empty() {
            return Err(ReplanningError::new(
                "replanning change list must not be empty",
            ));
        }

        let next_iteration = self.planning.replanning.iteration.saturating_add(1);
        for change in changes.clone() {
            self.apply_change(change)?;
        }
        self.planning.replanning.iteration = next_iteration;
        let attempt = ReplanningAttempt {
            iteration: next_iteration,
            reason: reason.into(),
            changes,
        };
        self.planning.replanning.history.push(attempt.clone());
        Ok(attempt)
    }

    fn apply_change(&mut self, change: TaskPlanningChange) -> Result<(), ReplanningError> {
        match change {
            TaskPlanningChange::AddTask { task } => {
                ensure_unique_task_id(&self.planning.planned_tasks, &task.id)?;
                self.planning.planned_tasks.push(task);
            }
            TaskPlanningChange::Split { from, into } => {
                let index = self.task_index(&from)?;
                if into.is_empty() {
                    return Err(ReplanningError::new("split must include replacement tasks"));
                }
                for task in &into {
                    if task.id == from {
                        return Err(ReplanningError::new(
                            "split replacement task id must differ from source task id",
                        ));
                    }
                    ensure_unique_task_id(&self.planning.planned_tasks, &task.id)?;
                }
                self.planning.planned_tasks[index].status = PlannedTaskStatus::Deferred;
                self.planning.planned_tasks[index].incomplete_reason =
                    Some(IncompleteReason::TooBroad);
                self.planning.planned_tasks[index].required = false;
                self.planning.planned_tasks.extend(into);
            }
            TaskPlanningChange::Replace { from, to } => {
                let index = self.task_index(&from)?;
                if to.id != from {
                    ensure_unique_task_id(&self.planning.planned_tasks, &to.id)?;
                }
                self.planning.planned_tasks[index] = to;
            }
            TaskPlanningChange::AddDependency {
                task_id,
                dependency_id,
            } => {
                self.task_index(&dependency_id)?;
                let index = self.task_index(&task_id)?;
                if !self.planning.planned_tasks[index]
                    .depends_on
                    .contains(&dependency_id)
                {
                    self.planning.planned_tasks[index]
                        .depends_on
                        .push(dependency_id);
                }
            }
            TaskPlanningChange::MarkBlocked { task_id, reason } => {
                let index = self.task_index(&task_id)?;
                self.planning.planned_tasks[index].status = PlannedTaskStatus::Blocked;
                self.planning.planned_tasks[index].incomplete_reason = Some(reason);
            }
            TaskPlanningChange::MarkCompleted { task_id, evidence } => {
                let index = self.task_index(&task_id)?;
                self.planning.planned_tasks[index].status = PlannedTaskStatus::Completed;
                self.planning.planned_tasks[index].incomplete_reason = None;
                self.planning.verification_evidence.extend(evidence);
            }
            TaskPlanningChange::Defer { task_id, reason: _ } => {
                let index = self.task_index(&task_id)?;
                self.planning.planned_tasks[index].status = PlannedTaskStatus::Deferred;
                self.planning.planned_tasks[index].required = false;
            }
        }
        Ok(())
    }

    fn task_index(&self, task_id: &str) -> Result<usize, ReplanningError> {
        self.planning
            .planned_tasks
            .iter()
            .position(|task| task.id == task_id)
            .ok_or_else(|| ReplanningError::new(format!("planned task not found: {task_id}")))
    }

    fn missing_required_criteria(&self) -> Vec<String> {
        if !self
            .planning
            .completion_policy
            .require_required_criteria_evidence
        {
            return Vec::new();
        }

        let passed_criteria = self
            .planning
            .verification_evidence
            .iter()
            .filter(|evidence| evidence.passed)
            .filter_map(|evidence| evidence.criterion_id.as_deref())
            .collect::<BTreeSet<_>>();

        self.planning
            .required_criteria()
            .filter(|criterion| !passed_criteria.contains(criterion.id.as_str()))
            .map(|criterion| criterion.id.clone())
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplanningProposal {
    pub reason: String,
    pub changes: Vec<TaskPlanningChange>,
}

impl ReplanningProposal {
    #[must_use]
    pub fn new(reason: impl Into<String>, changes: Vec<TaskPlanningChange>) -> Self {
        Self {
            reason: reason.into(),
            changes,
        }
    }
}

#[must_use]
pub fn planning_agent_system_contract(max_replanning_iterations: u8) -> String {
    format!(
        "# Planning and verification contract\n\
         - For PlanningAgent turns, maintain a TaskPlanning state with PlannedTask items.\n\
         - A PlannedTask is complete only when its required acceptance criteria have VerificationEvidence.\n\
         - If any required PlannedTask cannot complete, classify the IncompleteReason and replan it into smaller or unblocked PlannedTask items before finalizing.\n\
         - Run at most {max_replanning_iterations} replanning iterations. Stop earlier if user input is required.\n\
         - Do not silently drop unfinished work. Keep a replanning history that records split, replace, dependency, blocked, completed, or deferred changes.\n\
         - Final responses for incomplete work must state which PlannedTask items remain incomplete and why."
    )
}

fn ensure_unique_task_id(tasks: &[PlannedTask], task_id: &str) -> Result<(), ReplanningError> {
    if tasks.iter().any(|task| task.id == task_id) {
        Err(ReplanningError::new(format!(
            "planned task id already exists: {task_id}"
        )))
    } else {
        Ok(())
    }
}

#[must_use]
pub fn summarize_incomplete_reasons(planning: &TaskPlanning) -> BTreeMap<IncompleteReason, usize> {
    let mut counts = BTreeMap::new();
    for task in planning.incomplete_required_tasks() {
        let reason = task.incomplete_reason.unwrap_or(IncompleteReason::Unknown);
        *counts.entry(reason).or_insert(0) += 1;
    }
    counts
}

impl Ord for IncompleteReason {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (*self as u8).cmp(&(*other as u8))
    }
}

impl PartialOrd for IncompleteReason {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_planning() -> TaskPlanning {
        let mut planning = TaskPlanning::with_max_replanning_iterations("ship feature", 2);
        planning.planned_tasks.push(
            PlannedTask::new("task-1", "implement feature")
                .with_status(PlannedTaskStatus::Completed),
        );
        planning.planned_tasks.push(
            PlannedTask::new("task-2", "verify feature")
                .with_incomplete_reason(IncompleteReason::MissingDependency),
        );
        planning
            .acceptance_criteria
            .push(AcceptanceCriterion::required("criterion-1", "tests pass"));
        planning
    }

    #[test]
    fn complete_requires_tasks_and_required_criteria_evidence() {
        let mut planning = TaskPlanning::new("ship feature");
        planning.planned_tasks.push(
            PlannedTask::new("task-1", "implement feature")
                .with_status(PlannedTaskStatus::Completed),
        );
        planning
            .acceptance_criteria
            .push(AcceptanceCriterion::required("criterion-1", "tests pass"));
        planning
            .verification_evidence
            .push(VerificationEvidence::passed(
                "evidence-1",
                Some("task-1".to_string()),
                Some("criterion-1".to_string()),
                "cargo test passed",
            ));

        let coordinator = PlanningCoordinator::new(planning);
        let decision = coordinator.evaluate_completion();

        assert_eq!(decision.status, CompletionDecisionStatus::Complete);
        assert!(decision.incomplete_task_ids.is_empty());
        assert!(decision.missing_criterion_ids.is_empty());
    }

    #[test]
    fn incomplete_planning_requests_replanning_until_max_iterations() {
        let planning = sample_planning();
        let mut coordinator = PlanningCoordinator::new(planning);

        let first = coordinator.evaluate_completion();
        assert_eq!(first.status, CompletionDecisionStatus::NeedsReplanning);
        assert!(first.replanning_allowed);

        coordinator
            .apply_replanning(
                "add missing dependency",
                vec![TaskPlanningChange::AddTask {
                    task: PlannedTask::new("task-3", "prepare test fixture"),
                }],
            )
            .expect("first replanning should apply");
        coordinator
            .apply_replanning(
                "mark still incomplete",
                vec![TaskPlanningChange::MarkBlocked {
                    task_id: "task-2".to_string(),
                    reason: IncompleteReason::VerificationFailed,
                }],
            )
            .expect("second replanning should apply");

        let final_decision = coordinator.evaluate_completion();
        assert_eq!(
            final_decision.status,
            CompletionDecisionStatus::IncompleteMaxIterations
        );
        assert!(!final_decision.replanning_allowed);
    }

    #[test]
    fn replanning_loop_can_split_and_complete_tasks() {
        let mut planning = TaskPlanning::with_max_replanning_iterations("ship feature", 3);
        planning.planned_tasks.push(
            PlannedTask::new("task-1", "large feature task")
                .with_incomplete_reason(IncompleteReason::TooBroad),
        );
        planning
            .acceptance_criteria
            .push(AcceptanceCriterion::required("criterion-1", "tests pass"));
        let mut coordinator = PlanningCoordinator::new(planning);
        let mut calls = 0;

        let decision = coordinator
            .run_replanning_loop(|_planning, decision| {
                calls += 1;
                if calls == 1 {
                    assert_eq!(decision.status, CompletionDecisionStatus::NeedsReplanning);
                    Ok(ReplanningProposal::new(
                        "split broad task",
                        vec![TaskPlanningChange::Split {
                            from: "task-1".to_string(),
                            into: vec![PlannedTask::new("task-1a", "small implementation")],
                        }],
                    ))
                } else {
                    Ok(ReplanningProposal::new(
                        "complete replacement task",
                        vec![TaskPlanningChange::MarkCompleted {
                            task_id: "task-1a".to_string(),
                            evidence: vec![VerificationEvidence::passed(
                                "evidence-1",
                                Some("task-1a".to_string()),
                                Some("criterion-1".to_string()),
                                "cargo test passed",
                            )],
                        }],
                    ))
                }
            })
            .expect("loop should complete");

        assert_eq!(decision.status, CompletionDecisionStatus::Complete);
        assert_eq!(coordinator.planning().replanning.iteration, 2);
        assert_eq!(coordinator.planning().replanning.history.len(), 2);
    }

    #[test]
    fn user_input_required_stops_replanning_loop() {
        let mut planning = TaskPlanning::with_max_replanning_iterations("ship feature", 3);
        planning.planned_tasks.push(
            PlannedTask::new("task-1", "choose deployment target")
                .with_incomplete_reason(IncompleteReason::UserInputRequired),
        );
        let mut coordinator = PlanningCoordinator::new(planning);

        let decision = coordinator
            .run_replanning_loop(|_, _| panic!("replanning should not run"))
            .expect("evaluation should succeed");

        assert_eq!(decision.status, CompletionDecisionStatus::NeedsUserInput);
        assert_eq!(coordinator.planning().replanning.iteration, 0);
    }

    #[test]
    fn planning_contract_mentions_iteration_limit() {
        let contract = planning_agent_system_contract(5);
        assert!(contract.contains("at most 5 replanning iterations"));
        assert!(contract.contains("TaskPlanning"));
        assert!(contract.contains("PlannedTask"));
    }
}

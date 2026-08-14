/** Public proposer-artifact constructors and structural contracts. */
import * as Model from "./Model.ts"

export const ProposerArtifactTypeId: typeof Model.ProposerArtifactTypeId = Model.ProposerArtifactTypeId

export type ProposerArtifact = Model.ProposerArtifact
export type ProposerArtifactInput = Model.ProposerArtifactInput
export type ProposerComponent = Model.ProposerComponent
export type ProposerPlan = Model.ProposerPlan
export type ProposerTargetContract = Model.ProposerTargetContract
export type AutoregressiveProposerStage = Model.AutoregressiveProposerStage
export type AutoregressiveProposerState = Model.AutoregressiveProposerState
export type AutoregressiveProposerOutput = Model.AutoregressiveProposerOutput

export const artifact = Model.Speculation.artifact

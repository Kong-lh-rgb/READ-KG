import pydantic
import yaml

from .base import EnvYaml

class DatasetYaml(pydantic.BaseModel):
    name: str
    text_encoder_name: str
    question_analysis_emb_dir: str | None = None
    question_analysis_required: bool = False

class DDEYaml(pydantic.BaseModel):
    num_rounds: int
    num_reverse_rounds: int
    relation_aware: bool = False
    gate_hidden_dim: int | None = None
    gate_dropout: float = 0.0
    relation_mix: float = 0.5
    use_directional_gates: bool = True
    temperature: float = 1.0
    condition_source: str = 'question'
    analysis_mix: float = 0.7

class RetrieverYaml(pydantic.BaseModel):
    topic_pe: bool
    DDE_kwargs: DDEYaml
    llm_ref_kwargs: dict = pydantic.Field(default_factory=dict)
    structural_gate_kwargs: dict = pydantic.Field(default_factory=dict)

class OptimizerYaml(pydantic.BaseModel):
    lr: float

class EvalYaml(pydantic.BaseModel):
    k_list: str

class LossYaml(pydantic.BaseModel):
    contrastive_weight: float = 0.0
    contrastive_temperature: float = 0.07
    max_hard_negatives: int = 256

class RetrieverExpYaml(pydantic.BaseModel):
    num_epochs: int
    patience: int
    save_prefix: str
    max_train_edges: int | None = None

class RetrieverTrainYaml(pydantic.BaseModel):
    env: EnvYaml
    dataset: DatasetYaml
    retriever: RetrieverYaml
    optimizer: OptimizerYaml
    eval: EvalYaml
    loss: LossYaml = pydantic.Field(default_factory=LossYaml)
    train: RetrieverExpYaml

def load_yaml(config_file):
    with open(config_file) as f:
        yaml_data = yaml.load(f, Loader=yaml.loader.SafeLoader)

    task = yaml_data.pop('task')
    assert task == 'retriever'
    
    config = RetrieverTrainYaml(**yaml_data).model_dump()
    config['eval']['k_list'] = [
        int(k) for k in config['eval']['k_list'].split(',')]

    return config

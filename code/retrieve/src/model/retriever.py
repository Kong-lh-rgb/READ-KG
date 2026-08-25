import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

class PEConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr='mean')

    def forward(self, edge_index, x):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j


class WeightedPEConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')

    def forward(self, edge_index, x, edge_weight):
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        return x_j * edge_weight.unsqueeze(-1)


class RelationGate(nn.Module):
    def __init__(self, emb_size, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4 * emb_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, q_emb, relation_embs):
        if q_emb.dim() == 1:
            q_emb = q_emb.unsqueeze(0)
        q_expand = q_emb.expand(relation_embs.size(0), -1)
        gate_inputs = torch.cat(
            [
                q_expand,
                relation_embs,
                q_expand * relation_embs,
                torch.abs(q_expand - relation_embs)
            ],
            dim=-1
        )
        return self.net(gate_inputs).squeeze(-1)

class DDE(nn.Module):
    def __init__(
        self,
        num_rounds,
        num_reverse_rounds,
        emb_size=None,
        relation_aware=False,
        gate_hidden_dim=None,
        gate_dropout=0.0,
        relation_mix=0.5,
        use_directional_gates=True,
        temperature=1.0,
        condition_source='question',
        analysis_mix=0.7
    ):
        super().__init__()

        self.relation_aware = relation_aware
        self.relation_mix = relation_mix
        self.temperature = max(float(temperature), 1e-6)
        self.use_directional_gates = use_directional_gates
        self.condition_source = condition_source
        self.analysis_mix = min(max(float(analysis_mix), 0.0), 1.0)

        self.layers = nn.ModuleList()
        for _ in range(num_rounds):
            self.layers.append(PEConv())

        self.reverse_layers = nn.ModuleList()
        for _ in range(num_reverse_rounds):
            self.reverse_layers.append(PEConv())

        if self.relation_aware:
            if emb_size is None:
                raise ValueError('emb_size is required when relation_aware DDE is enabled.')
            hidden_dim = gate_hidden_dim or emb_size
            self.weighted_layers = nn.ModuleList(
                [WeightedPEConv() for _ in range(num_rounds)]
            )
            self.reverse_weighted_layers = nn.ModuleList(
                [WeightedPEConv() for _ in range(num_reverse_rounds)]
            )
            self.forward_gate = RelationGate(
                emb_size=emb_size,
                hidden_dim=hidden_dim,
                dropout=gate_dropout
            )
            if self.use_directional_gates:
                self.reverse_gate = RelationGate(
                    emb_size=emb_size,
                    hidden_dim=hidden_dim,
                    dropout=gate_dropout
                )
            else:
                self.reverse_gate = self.forward_gate

    def _compute_edge_weights(self, q_emb, edge_index, relation_embs, gate_module):
        edge_logits = gate_module(q_emb, relation_embs) / self.temperature
        return softmax(edge_logits, edge_index[1])

    def _mix_messages(self, mean_out, weighted_out, relation_gate=None):
        relation_mix = self.relation_mix
        if relation_gate is not None:
            relation_mix = relation_mix * relation_gate.to(mean_out.device).reshape(1)
        return (1 - relation_mix) * mean_out + relation_mix * weighted_out

    def _has_step_condition(self, analysis_step_embs, analysis_step_mask):
        if analysis_step_embs is None or analysis_step_embs.numel() == 0:
            return False
        if analysis_step_mask is None:
            return analysis_step_embs.size(0) > 0
        return analysis_step_mask.sum().item() > 0

    def _get_step_condition(self, analysis_step_embs, analysis_step_mask, layer_idx):
        if not self._has_step_condition(analysis_step_embs, analysis_step_mask):
            return None

        if analysis_step_mask is None:
            valid_indices = torch.arange(
                analysis_step_embs.size(0),
                device=analysis_step_embs.device
            )
        else:
            valid_indices = torch.nonzero(analysis_step_mask > 0, as_tuple=False).reshape(-1)

        if len(valid_indices) == 0:
            return None
        step_pos = valid_indices[min(layer_idx, len(valid_indices) - 1)]
        return analysis_step_embs[step_pos].unsqueeze(0)

    def _blend_condition(self, q_emb, analysis_condition, analysis_gate=None):
        if analysis_condition is None:
            return q_emb
        analysis_mix = self.analysis_mix
        if analysis_gate is not None:
            analysis_mix = analysis_mix * analysis_gate.to(q_emb.device).reshape(1)
        return F.normalize(
            (1 - analysis_mix) * q_emb + analysis_mix * analysis_condition,
            p=2,
            dim=-1
        )

    def _get_condition_emb(
        self,
        q_emb,
        analysis_emb,
        analysis_step_embs,
        analysis_step_mask,
        layer_idx,
        analysis_gate=None
    ):
        if q_emb.dim() == 1:
            q_emb = q_emb.unsqueeze(0)
        if analysis_emb is not None and analysis_emb.dim() == 1:
            analysis_emb = analysis_emb.unsqueeze(0)

        source = self.condition_source
        if source == 'question':
            return q_emb
        if source == 'analysis':
            return analysis_emb if analysis_emb is not None else q_emb
        if source == 'question_analysis':
            return self._blend_condition(q_emb, analysis_emb, analysis_gate)
        if source == 'steps':
            step_condition = self._get_step_condition(
                analysis_step_embs, analysis_step_mask, layer_idx
            )
            if step_condition is not None:
                return step_condition
            return analysis_emb if analysis_emb is not None else q_emb
        if source == 'question_steps':
            step_condition = self._get_step_condition(
                analysis_step_embs, analysis_step_mask, layer_idx
            )
            analysis_condition = step_condition if step_condition is not None else analysis_emb
            return self._blend_condition(q_emb, analysis_condition, analysis_gate)

        raise ValueError(f'Unknown DDE condition_source: {source}')

    def _compute_layer_edge_weights(
        self,
        q_emb,
        analysis_emb,
        analysis_step_embs,
        analysis_step_mask,
        layer_idx,
        edge_index,
        relation_embs,
        gate_module,
        analysis_gate=None
    ):
        condition_emb = self._get_condition_emb(
            q_emb,
            analysis_emb,
            analysis_step_embs,
            analysis_step_mask,
            layer_idx,
            analysis_gate=analysis_gate
        )
        return self._compute_edge_weights(condition_emb, edge_index, relation_embs, gate_module)
    
    def forward(
        self,
        topic_entity_one_hot,
        edge_index,
        reverse_edge_index,
        q_emb=None,
        r_id_tensor=None,
        relation_embs=None,
        analysis_emb=None,
        analysis_step_embs=None,
        analysis_step_mask=None,
        relation_gate=None,
        analysis_gate=None
    ):
        result_list = []

        if self.relation_aware:
            if q_emb is None or r_id_tensor is None or relation_embs is None:
                raise ValueError('q_emb, r_id_tensor, and relation_embs are required for relation-aware DDE.')
            edge_relation_embs = relation_embs[r_id_tensor]

        h_pe = topic_entity_one_hot
        for layer_idx, layer in enumerate(self.layers):
            mean_out = layer(edge_index, h_pe)
            if self.relation_aware:
                forward_edge_weight = self._compute_layer_edge_weights(
                    q_emb,
                    analysis_emb,
                    analysis_step_embs,
                    analysis_step_mask,
                    layer_idx,
                    edge_index,
                    edge_relation_embs,
                    self.forward_gate,
                    analysis_gate=analysis_gate
                )
                weighted_out = self.weighted_layers[layer_idx](
                    edge_index, h_pe, forward_edge_weight
                )
                h_pe = self._mix_messages(mean_out, weighted_out, relation_gate)
            else:
                h_pe = mean_out
            result_list.append(h_pe)

        h_pe_rev = topic_entity_one_hot
        for layer_idx, layer in enumerate(self.reverse_layers):
            mean_out = layer(reverse_edge_index, h_pe_rev)
            if self.relation_aware:
                reverse_edge_weight = self._compute_layer_edge_weights(
                    q_emb,
                    analysis_emb,
                    analysis_step_embs,
                    analysis_step_mask,
                    layer_idx,
                    reverse_edge_index,
                    edge_relation_embs,
                    self.reverse_gate,
                    analysis_gate=analysis_gate
                )
                weighted_out = self.reverse_weighted_layers[layer_idx](
                    reverse_edge_index, h_pe_rev, reverse_edge_weight
                )
                h_pe_rev = self._mix_messages(mean_out, weighted_out, relation_gate)
            else:
                h_pe_rev = mean_out
            result_list.append(h_pe_rev)

        return result_list

class Retriever(nn.Module):
    def __init__(
        self,
        emb_size,
        topic_pe,
        DDE_kwargs,
        llm_ref_kwargs=None,
        structural_gate_kwargs=None
    ):
        super().__init__()
        
        self.non_text_entity_emb = nn.Embedding(1, emb_size)
        self.topic_pe = topic_pe
        self.dde = DDE(emb_size=emb_size, **DDE_kwargs)
        self.llm_ref_kwargs = llm_ref_kwargs or {}
        self.use_llm_ref_features = bool(self.llm_ref_kwargs.get('enable', False))
        self.llm_ref_feature_dim = 7 if self.use_llm_ref_features else 0
        self.structural_gate_kwargs = structural_gate_kwargs or {}
        self.use_structural_gate = bool(self.structural_gate_kwargs.get('enable', False))
        
        pred_in_size = 4 * emb_size
        if topic_pe:
            pred_in_size += 2 * 2
        pred_in_size += 2 * 2 * (DDE_kwargs['num_rounds'] + DDE_kwargs['num_reverse_rounds'])
        pred_in_size += self.llm_ref_feature_dim

        self.pred = nn.Sequential(
            nn.Linear(pred_in_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, 1)
        )

    def _compute_structural_gate(self, analysis_step_mask, device):
        if not self.use_structural_gate:
            return None

        min_gate = float(self.structural_gate_kwargs.get('min_gate', 0.2))
        max_gate = float(self.structural_gate_kwargs.get('max_gate', 1.0))
        min_gate = max(0.0, min(min_gate, 1.0))
        max_gate = max(min_gate, min(max_gate, 1.0))
        threshold = float(self.structural_gate_kwargs.get('threshold', 1.5))
        slope = float(self.structural_gate_kwargs.get('slope', 4.0))

        if analysis_step_mask is None or analysis_step_mask.numel() == 0:
            num_steps = torch.tensor(0.0, device=device)
        else:
            num_steps = analysis_step_mask.to(device).float().sum()

        if self.structural_gate_kwargs.get('mode', 'sigmoid') == 'linear':
            target_steps = max(float(self.structural_gate_kwargs.get('target_steps', 2.0)), 1.0)
            gate_unit = torch.clamp((num_steps - 1.0) / max(target_steps - 1.0, 1e-6), 0.0, 1.0)
        else:
            gate_unit = torch.sigmoid(slope * (num_steps - threshold))

        return min_gate + (max_gate - min_gate) * gate_unit

    def _use_gate(self, key, fallback_key=None, default=True):
        if key in self.structural_gate_kwargs:
            return bool(self.structural_gate_kwargs.get(key))
        if fallback_key is not None and fallback_key in self.structural_gate_kwargs:
            return bool(self.structural_gate_kwargs.get(fallback_key))
        return default

    def _llm_ref_feature_indices_to_gate(self):
        feature_names = {
            'relation_step': 0,
            'relation_analysis': 1,
            'triple_step': 2,
            'triple_analysis': 3,
            'grounded_relation': 4,
            'endpoint_step': 5,
            'endpoint_analysis': 6,
        }
        configured = self.structural_gate_kwargs.get('llm_ref_gate_features', None)
        if configured is None:
            return list(range(self.llm_ref_feature_dim))
        indices = []
        for item in configured:
            if isinstance(item, int):
                idx = item
            else:
                idx = feature_names.get(str(item))
            if idx is not None and 0 <= idx < self.llm_ref_feature_dim:
                indices.append(idx)
        return sorted(set(indices))

    def _apply_llm_ref_gate(self, llm_ref_features, structural_gate):
        if structural_gate is None or not self._use_gate('apply_to_llm_ref', default=True):
            return llm_ref_features
        gated = llm_ref_features.clone()
        indices = self._llm_ref_feature_indices_to_gate()
        if indices:
            gated[:, indices] = gated[:, indices] * structural_gate.reshape(1, 1)
        return gated

    def _get_valid_step_embs(self, analysis_step_embs, analysis_step_mask):
        if analysis_step_embs is None or analysis_step_embs.numel() == 0:
            return None
        if analysis_step_mask is not None and analysis_step_mask.numel() > 0:
            valid = analysis_step_mask.reshape(-1) > 0
            analysis_step_embs = analysis_step_embs[valid]
        if analysis_step_embs.numel() == 0:
            return None
        return F.normalize(analysis_step_embs, p=2, dim=-1)

    def _compute_llm_ref_features(
        self,
        h_id_tensor,
        r_id_tensor,
        t_id_tensor,
        raw_entity_embs,
        relation_embs,
        analysis_emb,
        analysis_step_embs,
        analysis_step_mask
    ):
        num_triples = len(r_id_tensor)
        if not self.use_llm_ref_features or num_triples == 0:
            return relation_embs.new_zeros(num_triples, 0)

        device = relation_embs.device
        ref_features = relation_embs.new_zeros(num_triples, self.llm_ref_feature_dim)
        step_embs = self._get_valid_step_embs(analysis_step_embs, analysis_step_mask)
        analysis_condition = None
        if analysis_emb is not None and analysis_emb.numel() > 0:
            analysis_condition = F.normalize(analysis_emb.reshape(1, -1), p=2, dim=-1)

        h_raw = raw_entity_embs[h_id_tensor]
        r_raw = relation_embs[r_id_tensor]
        t_raw = raw_entity_embs[t_id_tensor]

        relation_weight = float(self.llm_ref_kwargs.get('relation_weight', 0.7))
        entity_weight = float(self.llm_ref_kwargs.get('entity_weight', 0.3))
        triple_vecs = relation_weight * r_raw + entity_weight * 0.5 * (h_raw + t_raw)
        h_norm = F.normalize(h_raw, p=2, dim=-1)
        r_norm = F.normalize(r_raw, p=2, dim=-1)
        t_norm = F.normalize(t_raw, p=2, dim=-1)
        triple_norm = F.normalize(triple_vecs, p=2, dim=-1)

        if step_embs is not None:
            rel_step_sims = r_norm @ step_embs.T
            triple_step_sims = triple_norm @ step_embs.T
            head_step_sims = h_norm @ step_embs.T
            tail_step_sims = t_norm @ step_embs.T
            endpoint_step_sims = torch.maximum(head_step_sims, tail_step_sims)
            ref_features[:, 0] = rel_step_sims.max(dim=1).values
            ref_features[:, 2] = triple_step_sims.max(dim=1).values
            ref_features[:, 5] = endpoint_step_sims.max(dim=1).values

            top_k_relations = int(self.llm_ref_kwargs.get('top_k_relations', 5))
            if top_k_relations > 0:
                unique_relation_ids = torch.unique(r_id_tensor)
                unique_rel_embs = F.normalize(
                    relation_embs[unique_relation_ids],
                    p=2,
                    dim=-1
                )
                unique_rel_step_sims = unique_rel_embs @ step_embs.T
                grounded_relation_mask = torch.zeros(
                    relation_embs.size(0),
                    device=device,
                    dtype=relation_embs.dtype
                )
                for step_idx in range(unique_rel_step_sims.size(1)):
                    k = min(top_k_relations, unique_rel_step_sims.size(0))
                    top_ids = torch.topk(
                        unique_rel_step_sims[:, step_idx],
                        k=k,
                        largest=True
                    ).indices
                    grounded_relation_mask[unique_relation_ids[top_ids]] = 1.
                ref_features[:, 4] = grounded_relation_mask[r_id_tensor]

        if analysis_condition is not None:
            ref_features[:, 1] = (r_norm @ analysis_condition.T).reshape(-1)
            ref_features[:, 3] = (triple_norm @ analysis_condition.T).reshape(-1)
            head_analysis_sim = (h_norm @ analysis_condition.T).reshape(-1)
            tail_analysis_sim = (t_norm @ analysis_condition.T).reshape(-1)
            ref_features[:, 6] = torch.maximum(head_analysis_sim, tail_analysis_sim)

        return ref_features

    def forward(
        self,
        h_id_tensor,
        r_id_tensor,
        t_id_tensor,
        q_emb,
        entity_embs,
        num_non_text_entities,
        relation_embs,
        topic_entity_one_hot,
        analysis_emb=None,
        analysis_step_embs=None,
        analysis_step_mask=None
    ):
        device = entity_embs.device
        
        raw_entity_embs = torch.cat(
            [
                entity_embs,
                self.non_text_entity_emb(
                    torch.LongTensor([0]).to(device)).expand(num_non_text_entities, -1)
            ]
        , dim=0)
        h_e_list = [raw_entity_embs]
        if self.topic_pe:
            h_e_list.append(topic_entity_one_hot)

        edge_index = torch.stack([
            h_id_tensor,
            t_id_tensor
        ], dim=0)
        reverse_edge_index = torch.stack([
            t_id_tensor,
            h_id_tensor
        ], dim=0)
        structural_gate = self._compute_structural_gate(analysis_step_mask, device)
        dde_relation_gate = structural_gate if self._use_gate(
            'apply_to_dde_relation',
            fallback_key='apply_to_dde',
            default=True
        ) else None
        dde_analysis_gate = structural_gate if self._use_gate(
            'apply_to_dde_analysis',
            fallback_key='apply_to_dde',
            default=True
        ) else None
        dde_list = self.dde(
            topic_entity_one_hot,
            edge_index,
            reverse_edge_index,
            q_emb=q_emb,
            r_id_tensor=r_id_tensor,
            relation_embs=relation_embs,
            analysis_emb=analysis_emb,
            analysis_step_embs=analysis_step_embs,
            analysis_step_mask=analysis_step_mask,
            relation_gate=dde_relation_gate,
            analysis_gate=dde_analysis_gate
        )
        h_e_list.extend(dde_list)
        h_e = torch.cat(h_e_list, dim=1)

        h_q = q_emb
        # Potentially memory-wise problematic
        h_r = relation_embs[r_id_tensor]

        h_triple = torch.cat([
            h_q.expand(len(h_r), -1),
            h_e[h_id_tensor],
            h_r,
            h_e[t_id_tensor]
        ], dim=1)
        if self.use_llm_ref_features:
            llm_ref_features = self._compute_llm_ref_features(
                h_id_tensor,
                r_id_tensor,
                t_id_tensor,
                raw_entity_embs,
                relation_embs,
                analysis_emb,
                analysis_step_embs,
                analysis_step_mask
            )
            llm_ref_features = self._apply_llm_ref_gate(llm_ref_features, structural_gate)
            h_triple = torch.cat([h_triple, llm_ref_features], dim=1)
        
        return self.pred(h_triple)

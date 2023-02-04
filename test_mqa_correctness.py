import torch
torch.manual_seed(0)

from transformers import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from transformers.models.gpt2.modeling_gpt2 import AttentionType

def get_attention(attention_type : AttentionType):
    config = GPT2Config.from_pretrained("gpt2")
    config.attention_type = attention_type.value
    return GPT2Attention(config)

def test_mqa_correctness():
    random_attn_weight = torch.randn(768, 768)
    random_attn_k_weight = torch.randn(768, 64)
    random_attn_v_weight = torch.randn(768, 64)
    random_attn_bias = torch.randn(768)
    random_attn_k_bias = torch.randn(64)
    random_attn_v_bias = torch.randn(64)
    random_proj = torch.randn(768, 768)
    random_proj_bias = torch.randn(768)

    # MULTI-HEAD ATTENTION
    c_attn_weight = torch.hstack(
        [random_attn_weight] + 
        12 * [random_attn_k_weight] +
        12 * [random_attn_v_weight])

    c_attn_bias = torch.hstack(
        [random_attn_bias] + 
        12 * [random_attn_k_bias] +
        12 * [random_attn_v_bias])

    attention_mh = get_attention(AttentionType.MULTI_HEAD)
    state_dict = attention_mh.state_dict()
    state_dict["c_attn.weight"] = c_attn_weight
    state_dict["c_attn.bias"] = c_attn_bias
    state_dict["c_proj.weight"] = random_proj
    state_dict["c_proj.bias"] = random_proj_bias
    attention_mh.load_state_dict(state_dict)

    # MULTI-QUERY ATTENTION 1
    attention_mq1 = get_attention(AttentionType.MULTI_QUERY_1)
    state_dict_mq1 = attention_mq1.state_dict()
    c_attn_weight = torch.hstack([random_attn_weight, random_attn_k_weight, random_attn_v_weight])
    c_attn_bias = torch.hstack([random_attn_bias, random_attn_k_bias, random_attn_v_bias])
    state_dict_mq1["c_attn.weight"] = c_attn_weight
    state_dict_mq1["c_attn.bias"] = c_attn_bias
    state_dict_mq1["c_proj.weight"] = random_proj
    state_dict_mq1["c_proj.bias"] = random_proj_bias 
    attention_mq1.load_state_dict(state_dict_mq1)

    # MULTI-QUERY ATTENTION 2
    attention_mq2 = get_attention(AttentionType.MULTI_QUERY_2)
    state_dict_mq2 = attention_mq2.state_dict()
    state_dict_mq2["q_attn.weight"] = random_attn_weight
    state_dict_mq2["q_attn.bias"] = random_attn_bias
    state_dict_mq2["kv_attn.weight"] = torch.hstack([random_attn_k_weight, random_attn_v_weight])
    state_dict_mq2["kv_attn.bias"] = torch.hstack([random_attn_k_bias, random_attn_v_bias])
    state_dict_mq2["c_proj.weight"] = random_proj
    state_dict_mq2["c_proj.bias"] = random_proj_bias
    attention_mq2.load_state_dict(state_dict_mq2)

    # Run correctness test
    attention_mh.eval()
    attention_mq1.eval()
    attention_mq2.eval()

    hidden_states = torch.randn(1, 5, 768)
    attention_mh_result = attention_mh(hidden_states)[0]
    attention_mq1_result = attention_mq1(hidden_states)[0]
    attention_mq2_result = attention_mq2(hidden_states)[0]

    assert torch.allclose(attention_mh_result, attention_mq1_result)
    assert torch.allclose(attention_mh_result, attention_mq2_result)
    assert torch.allclose(attention_mq1_result, attention_mq2_result)

    print("Correctness test passed!")

if __name__ == "__main__":
    test_mqa_correctness()
import torch
from src.data import KnownsDataset
from src.model import load_model
from src.utils import get_std_embeddings,find_subject_token_range
from src.trace import trace_important_states, total_aie_effects
from src.plot import plot_trace_heatmap

def compare_plots(fact,model):
  result = trace_important_states(fact)
  scores = result['resid_table'].detach().cpu()
  next_word = result['correct_next_word']
  low_score = result['min_prob']
  subj_index = find_subject_token_range(model.to_tokens(fact['prompt'],prepend_bos=True),fact['subject'])
  labels = model.to_str_tokens(fact['prompt'])[1:]
  print(scores)
  plot_trace_heatmap(scores, labels,next_word,low_score,subj_index, title="Impact of restoring Hidden States")

  scores = result['mlp_table'].detach().cpu()
  print(scores)
  plot_trace_heatmap(scores, labels,next_word,low_score,subj_index,color="Greens", title="Impact of restoring MLP States")


  scores = result['attn_table'].detach().cpu()
  print(scores)
  plot_trace_heatmap(scores, labels,next_word,low_score,subj_index,color="Reds", title="Impact of restoring Attn States")


def main():
    model = load_model("gpt2-large")
    
    data_dir = "data"
    known_facts = KnownsDataset(data_dir)

    if len(known_facts) == 0:
        print("Dataset is empty! Exiting.")
        return

    print("\n=== Computing noise level based on subject embeddings ===")
    std_value = get_std_embeddings(model, known_facts)
    noise_level = 3 * std_value
    print(f"Noise level set to: {noise_level:.4f}")

    custom_prompt = {
        "prompt": "Virat Kohli plays the sport of",
        "subject": "Virat K",
    }
    compare_plots(custom_prompt,model)

    
if __name__ == "__main__":
    main()

import torch
import torch.nn.functional as F

@torch.no_grad()
def generate(model, input_ids, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
    """
    Generación autoregresiva con soporte para Greedy, Top-K, Top-P y Temperatura.
    """
    model.eval()
    for _ in range(max_new_tokens):
        # Asegurar que no pasamos de max_seq_len (descontando los globales internamente)
        idx_cond = input_ids if input_ids.size(1) <= (model.config.max_seq_len - model.config.num_globals) else input_ids[:, -(model.config.max_seq_len - model.config.num_globals):]
        
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] # Tomar la distribución del último token
        
        if temperature == 0.0:
            # Greedy puro
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                # Mantener al menos un token
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
                
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
        input_ids = torch.cat((input_ids, next_token), dim=1)
        
    return input_ids

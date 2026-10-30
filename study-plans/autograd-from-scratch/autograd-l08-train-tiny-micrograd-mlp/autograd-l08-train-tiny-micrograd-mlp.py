import torch

def train_tiny_micrograd_mlp(inputs: torch.Tensor, targets: torch.Tensor, weights: list[torch.Tensor], biases: list[torch.Tensor], learning_rate: float, steps: int) -> tuple:
    """
    Returns final predictions, final loss, trained weights, trained biases, and pre-update loss history.
    """
    with torch.no_grad():
        trained_weights=[w.detach().clone() for w in weights]
        trained_bias=[b.detach().clone() for b in biases]
        loss_history: list[torch.Tensor] =[]

        def forward(x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
            activtions=[x]
            h=x
            for w,b in zip(trained_weights,trained_bias):
                h=torch.tanh(h@w.transpose(0,1)+b)
                activtions.append(h)

            preds=h.squeeze(-1)
            diff=preds - targets
            loss=torch.sum(diff*diff)
            return activtions,preds,loss
        batch_size=inputs.shape[0]
        for _ in range(steps):
            activations,preds,loss=forward(inputs)
            loss_history.append(loss)

            if batch_size==0:
                continue
            grad_h = (2.0 * (preds - targets)).unsqueeze(-1)
            for idx in range(len(trained_weights)-1,-1,-1):
                h_out=activations[idx+1]
                h_in=activations[idx]
                w=trained_weights[idx]
                dz=grad_h*(1.0 - h_out*h_out)
                grad_w = dz.transpose(0, 1) @ h_in
                grad_b = dz.sum(dim=0)
                if idx > 0:
                    grad_h = dz @ w

                trained_weights[idx] = w - learning_rate * grad_w
                trained_bias[idx] = trained_bias[idx] - learning_rate * grad_b
        _, final_preds, final_loss = forward(inputs)

        return (
            final_preds,
            final_loss,
            trained_weights,
            trained_bias,
            loss_history,
        )
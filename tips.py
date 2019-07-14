# pretrained models
# to load to part of your network
state = model.state_dict()
state.update(partial)
model.load_state_dict(state)

# to load part of existing model to your net
net.load_state_dict(saved, strict = False)

# print no of trainiable parameters
print("No of parameters: ", sum(p.numel() for p in net.parameters() if p.requires_grad))

# freeze some layers
for name, param in net.named_parameters():
  if 'fc' not in name:
    param.requires_grad = False

    
# weighted cross-entropy
def get_weight_matrix(output, label, void_pixels):
    num_zeros = (label == 0).sum().type(torch.FloatTensor)
    num_ones  = (label == 1).sum().type(torch.FloatTensor)
    total = num_zeros + num_ones
    assert total == np.prod(label.size())

    zero_wt = num_ones / total
    one_wt = num_zeros / total
    weights = torch.zeros(label.size())
    weights[label == 0] = zero_wt
    weights[label == 1] = one_wt
    if void_pixels is not None and void_pixels.sum() != 0:
        weights[void_pixels] = 0
        if len(np.unique(weights.numpy())) != 3:
            assert np.unique(weights.numpy())[1] > 0.5
            weights[weights > 0] = 1
            # np.save('error.npy', label.cpu().numpy())
            # np.save('void.npy', void_pixels.cpu().numpy())
        assert len(np.unique(weights.numpy())) == 3 or (len(np.unique(weights.numpy())) == 2 and weights.numpy().max() == 1), np.unique(weights.numpy())
    else:
        assert (len(np.unique(weights.numpy())) == 2 or num_zeros == num_ones), np.unique(weights.numpy())
    return weights

def cross_entropy_loss_torch_version(output, label, void_pixels = None, device = "cuda:0"):
    assert (output.size() == label.size())
    wts = get_weight_matrix(output, label, void_pixels)
    crit = BCEWithLogitsLoss(weight = wts).cuda(device)
    loss = crit(output, label)
    return loss

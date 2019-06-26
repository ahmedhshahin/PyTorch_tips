# pretrained models
# to load to part of your network
state = model.state_dict()
state.update(partial)
model.load_state_dict(state)

# to load part of existing model to your net
net.load_state_dict(saved, strict = False)

# print no of trainiable parameters
print("No of parameters: ", sum(p.numel() for p in net.parameters() if p.requires_grad))

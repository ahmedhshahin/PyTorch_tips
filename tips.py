# pretrained models
# to load to part of your network
state = model.state_dict()
state.update(partial)
model.load_state_dict(state)

# to load part of existing model to your net
net.load_state_dict(saved, strict = False)

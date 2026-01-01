# Stuff TODO

- [x] Finished task
- [ ] Unfinished task

- [ ] Add final loss and MAC to filename after finishing training
- [ ] Separate finished runs and unfinished runs in the log directory
- [ ] Do back to back encodeing + decoding and measure actual bpd
- [ ] For that: We need to have pixel by pixel model decoding
- [ ] Enable full arithmetic coding in all cases when training is finished
- [ ] Quad tree like splitting of trained image encoder + small finetune
- [ ] Fix torch.compile issues with broken graphs

## Plan of action for today:

- [ ] Add encode/decode validation after training is finished
- [ ] Refactor the lossless encode file to be cleaner -> should have these parts:
- if time allows:
  - [ ] make models modular to allow easy printing of layers
- [ ] fix test function output logs
- [ ] measure real MACs of trained models

# Gated Recurrent Unit(GRU)

So far, we have learned about how the LSTM cell uses different gates and how it solves the vanishing gradient problem of the RNN. 
But, as you may have noticed, the LSTM cell has too many parameters due to the presence of many gates and states.

This increases our training time. So, we introduce the Gated Recurrent Units (GRU) cell, which acts as a simplified version of the LSTM cell. 
Unlike the LSTM cell, the GRU cell has only two gates(reset and update gate) and one hidden state.

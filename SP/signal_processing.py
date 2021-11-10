from copy import copy

import numpy as np


class SP_Block:
    def __init__(self, action, inputs: int = 1, outputs: int = 1, **kwargs):
        self.action = action
        self.inputs = inputs
        self.outputs = outputs
        self.kwargs = kwargs

    def __call__(self, *args):
        return self.action(*args, **self.kwargs)

    def send_to(self, receivers):
        if self.outputs > 1:
            if isinstance(receivers, tuple):
                assert sum(r.inputs for r in receivers) <= self.outputs

                def out_func(*args):
                    mid: tuple = self(*args)
                    current_output = 0
                    output_signals = list()
                    for r in receivers:
                        current_output_signal = r(*mid[current_output: current_output + r.inputs])
                        if isinstance(current_output_signal, tuple):
                            output_signals.extend(current_output_signal)
                        else:
                            output_signals.append(current_output_signal)
                        current_output += r.inputs
                    output_signals.extend(mid[current_output:])
                    return output_signals

                return SP_Block(lambda *args: out_func(*args), inputs=self.inputs, outputs=sum(r.outputs for r in receivers))
            else:
                return SP_Block(lambda *args: (receivers(*self(*args))), inputs=self.inputs, outputs=receivers.outputs)
        else:
            return SP_Block(lambda *args: (receivers(self(*args))), inputs=self.inputs, outputs=receivers.outputs)

    def receive_from(self, senders):
        if isinstance(senders, tuple):
            assert sum(r.outputs for r in senders) <= self.inputs

            def out_func(*args):
                current_input = 0
                mid_signals = list()
                for s in senders:
                    current_mid_signal = s(*args[current_input: current_input + s.inputs])
                    if isinstance(current_mid_signal, tuple):
                        mid_signals.extend(current_mid_signal)
                    else:
                        mid_signals.append(current_mid_signal)
                    current_input += s.inputs
                mid_signals.extend(args[current_input:])
                return self(*mid_signals)

            return SP_Block(lambda *args: out_func(*args), inputs=sum(r.inputs for r in senders), outputs=self.outputs)
        else:
            return senders.send_to(self)


class SP_Wire(SP_Block):
    def __init__(self):
        super().__init__(lambda signal: signal)


class SP_Split(SP_Block):
     def __init__(self, branches: int):
         super().__init__(lambda signal: (signal, ) * branches, outputs=branches)


class SP_Arrange(SP_Block):
    def __init__(self, placess: tuple):
        def mix(*signals):
            output = [None] * len(placess)
            for idx, place in enumerate(placess):
                output[idx] = signals[place]
            return output

        super().__init__(mix, inputs=len(placess), outputs=len(placess))

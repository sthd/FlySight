class SP_Block:
    """
    A signal processing block performs an action on one or more signal(s) and returns one or more result signal(s).
    """

    def __init__(self, action, inputs: int = 1, outputs: int = 1, **kwargs):
        """
        Initializer
        :param action: a callable function to be performed on the input signals. Should be passed as positional argument.
        :param inputs: The number of input signals expected. Should be passed as positional argument.
        :param outputs: The number of output signals expected. Should be passed as positional argument.
        :param kwargs: Any auxiliary data needed for the action function. should be passed with the correct keywords for the functions passed.
        """
        self.action = action
        self.inputs = inputs
        self.outputs = outputs
        self.kwargs = kwargs

    def __call__(self, *args):
        """
        Pass signals to the block.
        Syntax: <SP_Block instance name>(signal1, signal2, ...)
        :param args: The input signal(s) to be passed.
        :return: The result signal(s)
        """
        return self.action(*args, **self.kwargs)

    def send_to(self, receivers):
        """
        Connect output(s) of self to other SP_Block(s).
        :param receivers: The other SP_Block(s) to connect to.
        :return: new SP_Block which performs self and then the receivers.
        """
        if self.outputs > 1:
            return self._multiple_outputs_send_to(receivers)
        else:
            return self._single_output_send_to(receivers)

    def _multiple_outputs_send_to(self, receivers):
        if isinstance(receivers, tuple):
            return self._send_to_multiple_receivers(receivers)
        else:
            return self._send_to_single_receiver(receivers)

    def _send_to_single_receiver(self, receivers):
        return SP_Block(lambda *args: (receivers(*self(*args))), inputs=self.inputs, outputs=receivers.outputs)

    def _send_to_multiple_receivers(self, receivers: tuple):
        self._check_connectivity_to(receivers)
        return SP_Block(self._chain_to(receivers), inputs=self.inputs, outputs=sum(r.outputs for r in receivers))

    def _chain_to(self, receivers: tuple):
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

        return out_func

    def _check_connectivity_to(self, receivers: tuple):
        assert sum(r.inputs for r in receivers) <= self.outputs

    def _single_output_send_to(self, receivers):
        return SP_Block(lambda *args: (receivers(self(*args))), inputs=self.inputs, outputs=receivers.outputs)

    def receive_from(self, senders):
        """
        Connect output(s) of other SP_Block(s) to self.
        :param senders: The other SP_Block(s) to connect to self.
        :return: new SP_Block which performs the senders and then self.
        """
        if isinstance(senders, tuple):
            return self._receive_from_multiple_senders(senders)
        else:
            return senders.send_to(self)

    def _receive_from_multiple_senders(self, senders: tuple):
        self._check_connectivity_from(senders)

        return SP_Block(self._chain_from(senders), inputs=sum(r.inputs for r in senders), outputs=self.outputs)

    def _chain_from(self, senders: tuple):
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

        return out_func

    def _check_connectivity_from(self, senders: tuple):
        assert sum(r.outputs for r in senders) <= self.inputs

    def __repr__(self):
        return self.action.__name__


class SP_Wire(SP_Block):
    """
    Passes a signal unchanged.
    """
    def __init__(self):
        def wire(signal):
            return signal

        super().__init__(wire)


class SP_Split(SP_Block):
    """
    Passes a single signal unchanged to multiple branches.
    """
    def __init__(self, branches: int):
        """
        Initializer
        :param branches: The amount of branches to split to.
        """
        def split(signal):
            return (signal,) * branches

        super().__init__(split, outputs=branches)


class SP_Arrange(SP_Block):
    """
    Rearranges multiple signals to a new given order.
    """
    def __init__(self, new_order: tuple):
        def rearrange(*signals):
            output = [None] * len(new_order)
            sig_spread = list()
            for sigs in signals:
                if isinstance(sigs, list):
                    sig_spread.extend(sigs)
                else:
                    sig_spread.append(sigs)
            for idx, place in enumerate(new_order):
                output[idx] = sig_spread[place]
            return output

        super().__init__(rearrange, inputs=len(new_order), outputs=len(new_order))

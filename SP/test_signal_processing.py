import unittest
from unittest import TestCase

import numpy as np

from SP.signal_processing import SP_Block, SP_Wire, SP_Split, SP_Arrange


class TestSP_Block(unittest.TestCase):
    def test(self):
        pass

    def test_single_input_single_output_cable(self):
        cable = SP_Block(lambda sig: sig)
        signal = np.array((1, 2, 3))
        self.assertListEqual(signal.tolist(), cable(signal).tolist())

    def test_single_input_single_output_negative(self):
        negative = SP_Block(lambda sig: -1 * sig)
        signal = np.array((1, 2, 3))
        self.assertListEqual([-1, -2, -3], negative(signal).tolist())

    def test_double_input_single_output_add(self):
        addition = SP_Block(np.add)
        signal = np.array((1, 2, 3))
        self.assertListEqual([2, 4, 6], addition(signal, signal).tolist())

    def test_single_input_double_output_plus_minus(self):
        plus_minus = SP_Block(lambda sig: (sig, -1 * sig))
        signal = np.array((1, 2, 3))
        self.assertListEqual(signal.tolist(), plus_minus(signal)[0].tolist())
        self.assertListEqual([-1, -2, -3], plus_minus(signal)[1].tolist())

    def test_wrong_number_of_input_signals(self):
        addition = SP_Block(np.add, inputs=2)
        signal = np.array((1, 2, 3))
        self.assertRaises(ValueError, lambda: addition(signal))

    def test_send_to_one_output(self):
        times_5 = SP_Block(lambda sig: 5 * sig)
        plus_5 = SP_Block(lambda sig: sig + 5)
        signal = np.array((1, 2, 3))
        t_5_p_5 = times_5.send_to(plus_5)
        p_5_t_5 = plus_5.send_to(times_5)
        self.assertListEqual([10, 15, 20], t_5_p_5(signal).tolist())
        self.assertListEqual([30, 35, 40], p_5_t_5(signal).tolist())

    def test_send_to_two_outputs(self):
        plus_minus = SP_Block(lambda sig: (sig, -1 * sig), outputs=2)
        addition = SP_Block(np.add, inputs=2)
        signal = np.array((1, 2, 3))
        zero_maker = plus_minus.send_to(addition)
        self.assertListEqual([0, 0, 0], zero_maker(signal).tolist())

    def test_kwargs(self):
        def mul_with_kernel(signal, kernel):
            return signal * kernel

        mult_with_array = SP_Block(mul_with_kernel, kernel=np.array((0, 1, .5)))
        signal = np.array((1, 2, 3))
        self.assertListEqual([0, 2, 1.5], mult_with_array(signal).tolist())

    def test_kwargs_with_send_to(self):
        def mul_with_kernel(signal, kernel):
            return signal * kernel

        mult_with_array = SP_Block(mul_with_kernel, kernel=np.array((0, 1, .5)))
        plus_5 = SP_Block(lambda sig: sig + 5)
        mwa_p_5 = mult_with_array.send_to(plus_5)
        signal = np.array((1, 2, 3))
        self.assertListEqual([5, 7, 6.5], mwa_p_5(signal).tolist())

    def test_send_to_two_outputs_to_two_blocks(self):
        plus_minus = SP_Block(lambda sig: (sig, -1 * sig), outputs=2)
        plus_5 = SP_Block(lambda sig: sig + 5)
        cable = SP_Block(lambda sig: sig)
        signal = np.array((1, 2, 3))
        pp5m = plus_minus.send_to((plus_5, cable))
        self.assertListEqual([6, 7, 8], pp5m(signal)[0].tolist())
        self.assertListEqual([-1, -2, -3], pp5m(signal)[1].tolist())

    def test_send_to_two_outputs_to_two_blocks_with_the_second_one_splitting(self):
        plus_minus = SP_Block(lambda sig: (sig, -1 * sig), outputs=2)
        plus_5 = SP_Block(lambda sig: sig + 5)
        signal = np.array((1, 2, 3))
        pp5m = plus_minus.send_to((plus_5, plus_minus))
        self.assertListEqual([6, 7, 8], pp5m(signal)[0].tolist())
        self.assertListEqual([-1, -2, -3], pp5m(signal)[1].tolist())
        self.assertListEqual([1, 2, 3], pp5m(signal)[2].tolist())

    def test_receive_from_one_input(self):
        times_5 = SP_Block(lambda sig: 5 * sig)
        plus_5 = SP_Block(lambda sig: sig + 5)
        signal = np.array((1, 2, 3))
        t_5_p_5 = plus_5.receive_from(times_5)
        p_5_t_5 = times_5.receive_from(plus_5)
        self.assertListEqual([10, 15, 20], t_5_p_5(signal).tolist())
        self.assertListEqual([30, 35, 40], p_5_t_5(signal).tolist())

    def test_receive_from_two_outputs(self):
        plus_minus = SP_Block(lambda sig: (sig, -1 * sig), outputs=2)
        addition = SP_Block(np.add, inputs=2)
        signal = np.array((1, 2, 3))
        zero_maker = addition.receive_from(plus_minus)
        self.assertListEqual([0, 0, 0], zero_maker(signal).tolist())

    def test_kwargs_with_receive_from(self):
        def mul_with_kernel(signal, kernel):
            return signal * kernel

        mult_with_array = SP_Block(mul_with_kernel, kernel=np.array((0, 1, .5)))
        plus_5 = SP_Block(lambda sig: sig + 5)
        mwa_p_5 = plus_5.receive_from(mult_with_array)
        signal = np.array((1, 2, 3))
        self.assertListEqual([5, 7, 6.5], mwa_p_5(signal).tolist())

    def test_receive_from_two_inputs_from_two_blocks(self):
        addition = SP_Block(np.add, inputs=2)
        plus_5 = SP_Block(lambda sig: sig + 5)
        cable = SP_Block(lambda sig: sig)
        signal = np.array((1, 2, 3))
        pp5m = addition.receive_from((plus_5, cable))
        self.assertListEqual([7, 9, 11], pp5m(signal, signal).tolist())

    def test_send_to_and_recieve_from_io_numbers(self):
        cross = SP_Block(lambda sig1, sig2: (sig2, sig1), inputs=2, outputs=2)
        addition = SP_Block(np.add, inputs=2)
        cross_add = cross.send_to(addition)
        self.assertEqual(1, cross_add.outputs)
        self.assertEqual(2, cross_add.inputs)

        add_add_cross = cross.receive_from((addition, addition))
        self.assertEqual(2, add_add_cross.outputs)
        self.assertEqual(4, add_add_cross.inputs)

    def test_send_to_two_outputs_to_one_single_output_block(self):
        plus_minus = SP_Block(lambda sig: (sig, -1 * sig), outputs=2)
        plus_5 = SP_Block(lambda sig: sig + 5)
        signal = np.array((1, 2, 3))
        pp5m = plus_minus.send_to((plus_5,))
        self.assertListEqual([6, 7, 8], pp5m(signal)[0].tolist())
        self.assertListEqual([-1, -2, -3], pp5m(signal)[1].tolist())

    def test_recieve_from_two_inputs_from_one_single_output_block(self):
        addition = SP_Block(np.add, inputs=2)
        plus_5 = SP_Block(lambda sig: sig + 5)
        signal = np.array((1, 2, 3))
        p5a = addition.receive_from((plus_5,))
        self.assertListEqual([7, 9, 11], p5a(signal, signal).tolist())


class TestSP_Wire(unittest.TestCase):
    def test_wire(self):
        wire = SP_Wire()
        signal = np.array((1, 2, 3))
        self.assertListEqual(signal.tolist(), wire(signal).tolist())


class TestSP_Split(unittest.TestCase):
    def test_2_split(self):
        wire = SP_Split(2)
        signal = np.array((1, 2, 3))
        self.assertListEqual(signal.tolist(), wire(signal)[0].tolist())
        self.assertListEqual(signal.tolist(), wire(signal)[1].tolist())

    def test_3_split(self):
        wire = SP_Split(3)
        signal = np.array((1, 2, 3))
        self.assertListEqual(signal.tolist(), wire(signal)[0].tolist())
        self.assertListEqual(signal.tolist(), wire(signal)[1].tolist())
        self.assertListEqual(signal.tolist(), wire(signal)[2].tolist())


class TestSP_Arrange(TestCase):
    def test_switch_2(self):
         cross = SP_Arrange((1, 0))
         signal1 = np.array((1, 2, 3))
         signal2 = np.array((0, 2, 0))
         self.assertListEqual(signal1.tolist(), cross(signal1, signal2)[1].tolist())
         self.assertListEqual(signal2.tolist(), cross(signal1, signal2)[0].tolist())

if __name__ == '__main__':
    unittest.main()

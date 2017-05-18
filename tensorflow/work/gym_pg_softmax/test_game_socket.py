import gym_sw1.trans.pb.action_pb2 as action_pb
import gym_sw1.trans.pb.state_pb2 as state_pb
from gym_sw1.trans.game_socket import GameSocket

game_socket = GameSocket('10.20.72.87', 5000);

reset_action = action_pb.Action()
reset_action.type = action_pb.Action.RESET
reset_action.value.extend([1, 2, 3])

first_result = game_socket.send_action(reset_action.SerializeToString())
second_result = game_socket.send_action(reset_action.SerializeToString())

print(first_result)
print(second_result)

state1 = state_pb.State()
state1.ParseFromString(first_result)

state2 = state_pb.State()
state2.ParseFromString(second_result)

print('first: {}', state1)
print('second: {}', state2)

game_socket.close()

print('down')
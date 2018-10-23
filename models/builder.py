from models.vision import GridVision
from models.state import GridState, ExternalState
from models.policy import GridPolicy
from models.language import InstructionLanguage


def build_policy(stack_frames=4, environment="grid_world"):
    '''
    params:
        environment - ["grid_world"]
    '''
    if environment != "grid_world":
        raise NotImplementedError()

    # Variables: depth = number of items; width; height
    # Variables: number of tokens (to fiks?); maks instruction length

    num_actions = 6
    vision = GridVision(depth=2, width=10*stack_frames, height=10)
    lang = InstructionLanguage(vocabulary_size=10, max_instruction_len=10)
    state = ExternalState(4, 4)
    
    return GridPolicy(vision, lang, state, num_actions=num_actions)

    

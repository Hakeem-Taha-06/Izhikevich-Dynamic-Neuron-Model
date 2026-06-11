# This file is a leftover from the old Hodgkin-Huxley model.
# It has been superseded by the correct Izhikevich implementation at:
#     src/numerical/backward_euler.py   (Role 6)
#
# DO NOT import from this file. Use:
#     from src.numerical.backward_euler import solve_backward_euler
raise ImportError(
    "This file belongs to the deprecated Hodgkin-Huxley model. "
    "Use src/numerical/backward_euler.py for the Izhikevich solver."
)

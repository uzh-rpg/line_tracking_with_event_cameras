# A Line
int64 id           # unique line id

# states
uint8 INITIALIZING = 0
uint8 HIBERNATING = 1
uint8 ACTIVE = 2
uint8 state

float64 length      # length of line [px]

float64 pos_x       # x coordinate of midpoint [px]
float64 pos_y       # y coordinate of midpoint [px]
float64 theta       # angle wrt to y axis [rad]

float64 vel_x       # x component of velocity of midpoint [px/s]
float64 vel_y       # y component of velocity of midpoint [px/s]

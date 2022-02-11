# A Line
int64 id           # unique line id

# states
uint8 INITIALIZING = 0
uint8 HIBERNATING = 1
uint8 ACTIVE = 2
uint8 state

float64 length          # length of line [px]
float64 p_pos_x         # pixel x coordinate of midpoint [px]
float64 p_pos_y         # pixel y coordinate of midpoint [py]
float64 c_pos_x_end_1   # world x coordianate of midpoint
float64 c_pos_y_end_1   # world y coordianate of midpoint
float64 c_pos_z_end_1   # world z coordianate of midpoint
float64 c_pos_x_end_2   # world x coordianate of midpoint
float64 c_pos_y_end_2   # world y coordianate of midpoint
float64 c_pos_z_end_2   # world y coordianate of midpoint
float64 theta           # angle wrt to y axis [rad]
float64 vel_x           # x component of velocity of midpoint [px/s]
float64 vel_y           # y component of velocity of midpoint [px/s]

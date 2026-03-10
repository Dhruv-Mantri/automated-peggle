import pyautogui
import cv2
import numpy as np
import mss # fast screenshots
import pygetwindow as gw
import time
import random

# ML
import torch
import torch.nn as nn
import torch.nn.functional as nnf

windows = gw.getAllTitles()

peggle_window = gw.getWindowsWithTitle('Peggle Deluxe 1.01')[0]
left, top, width, height = peggle_window.left, peggle_window.top, peggle_window.width, peggle_window.height

print(width, height)


'''shoot ball at angle (deg)'''
def shoot_ball(angle):
    r = 200 # constant radius
    cx, cy = 604, 100 # cannon pos

    x = cx + r * np.sin(np.deg2rad(angle))
    y = cy + r * np.cos(np.deg2rad(angle))

    pyautogui.moveTo(x + left, y + top)
    pyautogui.click()


'''Get image of current screen'''
def current_state():
    with mss.mss() as sct:
        monitor = {
            "left": left+150,
            "top": top + 100,
            "width": width-300,
            "height": height - 150
        }

        # circle detection
        img = np.array(sct.grab(monitor))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    return img

# gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
# gray = cv2.medianBlur(gray, 5)

# circles = cv2.HoughCircles(
#     gray,
#     cv2.HOUGH_GRADIENT,
#     dp=1,
#     minDist=30,
#     param1=100,
#     param2=20,
#     minRadius=10,
#     maxRadius=25
# )
#
# if circles is not None:
#     for x, y, r in circles[0]:
#         if (int(y) > 150) and (int(x) > 150 and int(x) < width - 150):
#             cv2.circle(img, (int(x), int(y)), int(r), (0, 255, 0), 2)

orange_pegs = [0]
def update_pegs(img):
    global orange_pegs
    orange_pegs.clear()

    # detect colors
    hsv_frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # orange peg mask
    peg_lower_bound = np.array([0, 103, 160], np.uint8)
    peg_upper_bound = np.array([10, 237, 255], np.uint8) # mask is perfect for circle pegs, area 225
    peg_mask = cv2.inRange(hsv_frame, peg_lower_bound, peg_upper_bound)

    # Morphological transformations
    kernel = np.ones((5, 5), "uint8")
    peg_mask = cv2.dilate(peg_mask, kernel)

    # green peg mask
    green_peg_lower_bound = np.array([40, 81, 75], np.uint8)
    green_peg_upper_bound = np.array([62, 255, 213], np.uint8)
    # purple peg mask
    # blue peg mask? (optional)
    blue_peg_lower_bound = np.array([108, 119, 140], np.uint8)
    blue_peg_upper_bound = np.array([122, 160, 255], np.uint8)


    contours, _ = cv2.findContours(peg_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
       area = cv2.contourArea(contour)
       if area > 120 and area < 550: # Filter small areas
           x, y, w, h = cv2.boundingRect(contour)
           if ( (x > 360 and x < 545) and (y < 160) ):
               img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
               continue
           orange_pegs.append((x,y))
           img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
           # cv2.putText(img, "*"+str(area), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
           # cv2.imshow("Peggle View", img)
           # cv2.imshow("Red View", peg_mask)

    # print(len(orange_pegs))
    return peg_mask

'''Returns boolean whether there is an active ball in play'''
def active_ball(img):
    hsv_frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ball_lower_bound = np.array([18, 162, 117], np.uint8) #    18, 152, 109 :: 18, 162, 117 :: 2, 205, 220
    ball_upper_bound = np.array([34, 255, 164], np.uint8) #  24, 255, 162 :: 34, 255, 164 :: 27, 255, 255

    ball_mask = cv2.inRange(hsv_frame, ball_lower_bound, ball_upper_bound)

    # Morphological transformations
    kernel = np.ones((5, 5), "uint8")
    ball_mask = cv2.dilate(ball_mask, kernel)

    contours, _ = cv2.findContours(ball_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)

        x, y, w, h = cv2.boundingRect(contour)
        # return True, img, ball_mask
        # if (not (x > 360 and x < 545) and (y < 160)):
        if (area > 100 and w < 20 and h < 15):
            # cv2.putText(img, "*" + str(area) + str(w) + str(h), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255))
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
            return True#, img, ball_mask


    return False#, img, ball_mask

# function to test if active_ball() is working
# def test_active_ball():
#     time.sleep(5)
#     shoot_ball(40)
#     while True:
#         img = current_state()
#         print(active_ball(img))
#
# test_active_ball()


# primitive run method, just shoot at the first orange in list
def run():
    while (len(orange_pegs) > 0):
        img = current_state()
        update_pegs(img)
        time.sleep(1)
        # cv2.imshow("Peggle View", img)

        # calculate the ball with highest y pos
        shoot_ball(orange_pegs[0][0], orange_pegs[0][1])
        # print(orange_pegs[0][0], orange_pegs[0][1])
        while (active_ball(img) == True):
            # print(active_ball(img))
            img = current_state()
            time.sleep(5)

'''Test active_ball()'''
# img = current_state()
# ballornah, ball, ball_mask = active_ball(img)
# peg_mask = update_pegs(img)
#
# print(ballornah)
# cv2.imshow("Peggle View", img)
# cv2.imshow("Red View", peg_mask)
# cv2.imshow("Ball View", ball)
# cv2.imshow("BallMask View", ball_mask)
# cv2.waitKey(0)

def replay_level():
    time.sleep(5)
    pyautogui.moveTo(left + 739, top + 593) # restart button
    pyautogui.click()
    time.sleep(5)
    pyautogui.moveTo(left + 710, top + 760) # start level button
    pyautogui.click()
    time.sleep(2)

# REINFORCEMENT LEARNING
ACTIONS = np.linspace(-85, 85, 171)


# define the state
'''
# of orange pegs
y level of the pegs
ADD LATER
time in play
hit purple
free ball
end level with bonus balls
'''
'''
def get_state(): # Redefine get_state to take in image and determine all orange peg locations
    if not orange_pegs:
        return np.zeros(3, dtype=np.float32)

    ys = [y for _, y in orange_pegs]

    return np.array([
        len(orange_pegs), # less is better
        np.mean(ys),# bigger is better
        np.std(ys)
    ], dtype=np.float32)
    
'''

def get_state(img):
    resized = cv2.resize(img, (84, 84))
    normalized = resized.astype(np.float32) / 255.0

    # Transpose from (Height, Width, Channels) to (Channels, Height, Width)
    state = np.transpose(normalized, (2, 0, 1))

    return state


class PegglePlayer(nn.Module):
    def __init__(self):
        super().__init__()
        # self.net = nn.Sequential(
        #     nn.Linear(3, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, len(ACTIONS)),
        #     nn.Softmax(dim=-1)
        # )
        # Convert to CNN

        # 3 conv layers --> 84x84 to 7x7
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        # flatten (64 * 7 * 7 = 3136)
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, len(ACTIONS))



    def forward(self, x):
        # Pass through convs with relu
        x = nnf.relu(self.conv1(x))
        x = nnf.relu(self.conv2(x))
        x = nnf.relu(self.conv3(x))

        x = x.reshape(x.size(0), -1) # flatten

        # Pass through linear layers
        x = nnf.relu(self.fc1(x))
        x = self.fc2(x)

        # Output probs for each angle
        return nnf.softmax(x, dim=-1)

player = PegglePlayer()
optimizer = torch.optim.Adam(player.parameters(), lr=1e-3)

EPISODES = 500
for episode in range(EPISODES):
    replay_level()
    while True:

        img = current_state()
        update_pegs(img)

        if len(orange_pegs) == 0:
            break

        state = get_state(img)
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        probs = player(state_t)
        dist = torch.distributions.Categorical(probs)

        if random.random() < 0.2:
            action_idx = random.randrange(len(ACTIONS))
            log_prob = torch.log(probs[0, action_idx])
        else:
            action = dist.sample()
            action_idx = action.item()
            log_prob = dist.log_prob(action)

        pegs_before = len(orange_pegs)

        shoot_ball(ACTIONS[action_idx])
        while (active_ball(img) == True):
            img = current_state()
            time.sleep(2.5)
            print("Active Ball")

        img = current_state()
        update_pegs(img)
        pegs_after = len(orange_pegs)

        reward = pegs_before - pegs_after
        if reward == 0:
            reward = -5
        loss = -log_prob * reward

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Angle {ACTIONS[action_idx]:.1f}° | Reward {reward}")

    torch.save(player.state_dict(), "peggle_player.pt")
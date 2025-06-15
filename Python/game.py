import pygame
import sys
import random
import os
import numpy as np
import pickle

def run_pong_game(
    player_events=[],
    player_encoding=[],
    width=64,
    height=64,
    ball_speed_x=1,
    ball_speed_y=1,
    paddle_speed=2,
    max_score=5,
    frame_dir='pong_frames/pong',
):

    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Two Player Pong")

    bcd_config = {
    "segment_length": 2,
    "segment_thickness": 1,
    "digit_spacing": 0,
    "digit_color_on": (255, 255, 255),
    "digit_color_off": (0, 0, 0),
    "left_score_pos": (width//2 - 1 - 2 - 5, 2),
    "right_score_pos": (width//2 + 1 + 2, 2),
    "score_left": 0,
    "score_right": 0
    }

    # 7-segment definitions: [a, b, c, d, e, f, g]
    bcd_segments = {
        '0': [1, 1, 1, 1, 1, 1, 0],
        '1': [0, 1, 1, 0, 0, 0, 0],
        '2': [1, 1, 0, 1, 1, 0, 1],
        '3': [1, 1, 1, 1, 0, 0, 1],
        '4': [0, 1, 1, 0, 0, 1, 1],
        '5': [1, 0, 1, 1, 0, 1, 1],
        '6': [1, 0, 1, 1, 1, 1, 1],
        '7': [1, 1, 1, 0, 0, 0, 0],
        '8': [1, 1, 1, 1, 1, 1, 1],
        '9': [1, 1, 1, 1, 0, 1, 1]
    }

    def draw_segment(surface, x, y, length, thickness, on, orientation):
        color = bcd_config['digit_color_on'] if on else bcd_config['digit_color_off']
        if orientation == 'h':
            pygame.draw.rect(surface, color, (x, y, length, thickness))
        else:
            pygame.draw.rect(surface, color, (x, y, thickness, length))

    def draw_digit(surface, digit, x, y):
        seg = bcd_segments[str(digit)]
        l = bcd_config["segment_length"]
        t = bcd_config["segment_thickness"]
        
        # Segment positions
        draw_segment(surface, x + t, y, l, t, seg[0], 'h')                    # a
        draw_segment(surface, x + l + t, y + t, l, t, seg[1], 'v')            # b
        draw_segment(surface, x + l + t, y + l + 2*t, l, t, seg[2], 'v')      # c
        draw_segment(surface, x + t, y + 2*l + 2*t, l, t, seg[3], 'h')        # d
        draw_segment(surface, x, y + l + 2*t, l, t, seg[4], 'v')              # e
        draw_segment(surface, x, y + t, l, t, seg[5], 'v')                    # f
        draw_segment(surface, x + t, y + l + t, l, t, seg[6], 'h')            # g

    def draw_score(surface, score, pos):
        x, y = pos
        score_str = str(score)
        for digit in score_str:
            draw_digit(surface, digit, x, y)
            digit_width = bcd_config["segment_length"] + bcd_config["segment_thickness"]
            x += digit_width + bcd_config["digit_spacing"]

    # Colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (219, 68, 55)
    BLUE = (66, 133, 244)
    GREEN = (15, 157, 88)

    # Paddles and ball
    paddle1 = pygame.Rect(4, height//2 - 8, 2, 16)
    paddle2 = pygame.Rect(width - 4 - 2, height//2 - 8, 2, 16)
    ball = pygame.Rect(width//2 - 2, height//2 - 2, 5, 5)

    # Key states
    keys_held = {
        'w': False,
        's': False,
        'up': False,
        'down': False
    }

    # Scores
    score1 = 0
    score2 = 0

    # Logging setup
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir)

    with open(os.path.join(frame_dir, "player_encoding.pkl"), "wb") as f:
        pickle.dump(player_encoding, f)
        
    frame_count = 0

    def reset_ball():
        ball.center = (width//2, height//2 + 11)

    clock = pygame.time.Clock()
    score1, score2 = 0, 0
    ball_dx, ball_dy = ball_speed_x, ball_speed_y
    ball_reset = True
    for player_event in player_events:
        for event in player_event:
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    keys_held['w'] = True
                if event.key == pygame.K_s:
                    keys_held['s'] = True
                if event.key == pygame.K_UP:
                    keys_held['up'] = True
                if event.key == pygame.K_DOWN:
                    keys_held['down'] = True

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_w:
                    keys_held['w'] = False
                if event.key == pygame.K_s:
                    keys_held['s'] = False
                if event.key == pygame.K_UP:
                    keys_held['up'] = False
                if event.key == pygame.K_DOWN:
                    keys_held['down'] = False

        # Paddle 1 movement
        if keys_held['w'] and not keys_held['s']:
            paddle1_speed = -paddle_speed
        elif keys_held['s'] and not keys_held['w']:
            paddle1_speed = paddle_speed
        else:
            paddle1_speed = 0

        # Paddle 2 movement
        if keys_held['up'] and not keys_held['down']:
            paddle2_speed = -paddle_speed
        elif keys_held['down'] and not keys_held['up']:
            paddle2_speed = paddle_speed
        else:
            paddle2_speed = 0

        paddle1.y += paddle1_speed
        paddle2.y += paddle2_speed

        paddle1.y = max(min(paddle1.y, height - paddle1.height), 11)
        paddle2.y = max(min(paddle2.y, height - paddle2.height), 11)

        if not ball_reset:
            ball.x += ball_dx
            ball.y += ball_dy
        else:
            ball_reset = False

        if (ball.top <= 11 or ball.bottom >= height) and not (ball.left <= 0 or ball.right >= width):
            ball_dy *= -1

        if ball.colliderect(paddle1) and ball_dx < 0:
            if abs(ball.left - paddle1.right) < 2:
                ball_dx *= -1
            elif abs(ball.bottom - paddle1.top) < 2 or abs(ball.top - paddle1.bottom) < 2:
                ball_dy *= -1

        if ball.colliderect(paddle2) and ball_dx > 0:
            if abs(ball.right - paddle2.left) < 2:
                ball_dx *= -1
            elif abs(ball.bottom - paddle2.top) < 2 or abs(ball.top - paddle2.bottom) < 2:
                ball_dy *= -1

        if ball.left <= 0:
            score2 += 1
            ball_reset = True
            reset_ball()
        if ball.right >= width:
            score1 += 1
            ball_reset = True
            reset_ball()

        
        if score1 >= max_score or score2 >= max_score:
            score1 = 0
            score2 = 0
            
        # Start record after the zero black frame
        if frame_count >= 1:
            pygame.image.save(screen, f'{frame_dir}/frame_{frame_count}.png')

        screen.fill(BLACK)

        for y in range(10, height, 4):
            pygame.draw.line(screen, WHITE, (width // 2 - 1, y), (width // 2 - 1, y + 1), 1)
        
        pygame.draw.line(screen, WHITE, (0, 10), (width, 10), 1)

        pygame.draw.rect(screen, BLUE, paddle1)
        pygame.draw.rect(screen, RED, paddle2)

        if not ball_reset:
            pygame.draw.ellipse(screen, GREEN, ball)
        else:
            pass

        bcd_config["score_left"] = score1
        bcd_config["score_right"] = score2
        draw_score(screen, bcd_config["score_left"], bcd_config["left_score_pos"])
        draw_score(screen, bcd_config["score_right"], bcd_config["right_score_pos"])
            
        pygame.display.flip()
        clock.tick(60)
        frame_count += 1
    
    pygame.quit()
    
def generate_random_event_list(previous_button_logics, change_prob=0.05):
    # Define possible event types and keys that the game uses
    possible_keys = [pygame.K_w, pygame.K_s, pygame.K_UP, pygame.K_DOWN]
    possible_types = [pygame.KEYDOWN, pygame.KEYUP]

    flip_mask = np.random.rand(len(previous_button_logics)) < change_prob
    current_button_logics = list(np.array(previous_button_logics) ^ flip_mask.astype(int))
    
    event_encoding = []
    events = []
    for i, j, key in zip(previous_button_logics, current_button_logics, possible_keys):
        if i == 0 and j == 0:
            pass
        elif i == 0 and j == 1:
            event = pygame.event.Event(pygame.KEYDOWN, {'key': key})
            events.append(event)
        elif i == 1 and j == 0:
            event = pygame.event.Event(pygame.KEYUP, {'key': key})
            events.append(event)
        else: #i == 1 j == 1:
            pass

        event_encoding.extend([i, j])
        
    return events, event_encoding, current_button_logics

def generate_player_events(num_event_list):
    player_events = [[]] # No event in the zero frame (The zero frame didn't be saved by the game)
    player_encoding = []
    current_button_logics = [0, 0, 0 ,0 ,0]
    for i in range(num_event_list):
        events, event_encoding, current_button_logics = generate_random_event_list(current_button_logics)
        player_events.append(events)
        player_encoding.append(event_encoding)
        
    return player_events, player_encoding

if __name__ == "__main__":
    player_events, player_encoding = generate_player_events(100)
    run_pong_game(player_events, player_encoding)
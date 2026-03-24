import sys
import random
import pygame

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCREEN_W      = 400
SCREEN_H      = 600
FPS           = 60
TITLE         = "Flappy Poodle"

GRAVITY       = 0.5
FLAP_STRENGTH = -9
MAX_FALL      = 10

PIPE_WIDTH    = 60
PIPE_GAP      = 160
PIPE_SPEED    = 3
PIPE_SPACING  = 220
PIPE_CAP_H    = 16
PIPE_CAP_EXTRA = 4

BIRD_W        = 40
BIRD_H        = 30
BIRD_X        = 80

GROUND_H      = 80

SKY_COLOR     = (100, 180, 230)
GROUND_COLOR  = (210, 170, 90)
GROUND_LINE   = (180, 140, 60)
PIPE_COLOR    = (80, 160, 80)
PIPE_DARK     = (60, 130, 60)
POODLE_BLACK  = (15, 15, 15)
POODLE_DARK   = (45, 45, 45)
WING_COLOR    = (220, 235, 255)
WING_DARK     = (170, 195, 230)
NOSE_COLOR    = (200, 90, 90)
TEXT_COLOR    = (255, 255, 255)
SHADOW_COLOR  = (0, 0, 0)
OVERLAY_COLOR = (0, 0, 0, 160)

# Maltipoo palette
MALTIPOO_LIGHT = (232, 168, 92)
MALTIPOO_DARK  = (200, 136, 74)
MALTIPOO_EAR   = (184, 116, 62)
MALTIPOO_NOSE  = (26, 16, 16)
MALTIPOO_NOSE2 = (45, 26, 10)


# ---------------------------------------------------------------------------
# Sprite factories (module-level, called after pygame.init)
# ---------------------------------------------------------------------------
def make_poodle_surface():
    """Black toy poodle — Pillow."""
    surf = pygame.Surface((BIRD_W, BIRD_H), pygame.SRCALPHA)
    left_wing  = [(6, 17), (15, 17), (12, 3), (3, 6)]
    right_wing = [(19, 17), (28, 17), (32, 6), (22, 3)]
    for wing in (left_wing, right_wing):
        pygame.draw.polygon(surf, WING_DARK, wing)
        pygame.draw.polygon(surf, WING_COLOR, [(x + 1, y + 1) for x, y in wing])
    pygame.draw.circle(surf, POODLE_DARK,  (4, 20), 6)
    pygame.draw.circle(surf, POODLE_BLACK, (3, 19), 5)
    pygame.draw.ellipse(surf, POODLE_DARK,  (3, 12, 26, 14))
    pygame.draw.ellipse(surf, POODLE_BLACK, (2, 11, 25, 13))
    hx, hy = 29, 16
    for dx, dy in [(7,0),(5,5),(0,7),(-5,5),(-7,0),(-5,-5),(0,-7),(5,-5)]:
        pygame.draw.circle(surf, POODLE_BLACK, (hx + dx, hy + dy), 4)
    pygame.draw.circle(surf, POODLE_BLACK, (hx, hy), 6)
    pygame.draw.circle(surf, (255, 255, 255), (31, 11), 4)
    pygame.draw.circle(surf, (10, 10, 10),    (32, 11), 2)
    pygame.draw.circle(surf, (255, 255, 255), (33, 10), 1)
    pygame.draw.circle(surf, NOSE_COLOR,    (37, 17), 3)
    pygame.draw.circle(surf, (150, 50, 50), (37, 17), 2)
    return surf


def make_maltipoo_surface():
    """Apricot maltipoo — Blanket."""
    surf = pygame.Surface((BIRD_W, BIRD_H), pygame.SRCALPHA)
    left_wing  = [(6, 17), (15, 17), (12, 3), (3, 6)]
    right_wing = [(19, 17), (28, 17), (32, 6), (22, 3)]
    for wing in (left_wing, right_wing):
        pygame.draw.polygon(surf, WING_DARK, wing)
        pygame.draw.polygon(surf, WING_COLOR, [(x + 1, y + 1) for x, y in wing])
    # Floppy drop ears (drawn before body)
    pygame.draw.ellipse(surf, MALTIPOO_EAR, (20, 14, 10, 18))
    pygame.draw.ellipse(surf, MALTIPOO_EAR, (28, 13,  9, 17))
    # Tail — soft oval
    pygame.draw.ellipse(surf, MALTIPOO_DARK,  (0, 15, 10, 8))
    pygame.draw.ellipse(surf, MALTIPOO_LIGHT, (1, 16,  8, 6))
    # Body
    pygame.draw.ellipse(surf, MALTIPOO_DARK,  (3, 12, 26, 14))
    pygame.draw.ellipse(surf, MALTIPOO_LIGHT, (2, 11, 25, 13))
    # Head fluff — loose wavy puffs
    hx, hy = 29, 16
    for dx, dy in [(-7,-1),(-5,-5),(-1,-7),(3,-6),(7,-3),(8,1),(6,5)]:
        pygame.draw.circle(surf, MALTIPOO_DARK, (hx + dx, hy + dy), 5)
    # Head face — rounder
    pygame.draw.circle(surf, MALTIPOO_LIGHT, (hx, hy), 7)
    # Eye
    pygame.draw.circle(surf, (255, 255, 255), (31, 11), 4)
    pygame.draw.circle(surf, (58, 31, 8),     (32, 11), 2)   # dark brown
    pygame.draw.circle(surf, (255, 255, 255), (33, 10), 1)
    # Nose — dark
    pygame.draw.ellipse(surf, MALTIPOO_NOSE,  (33, 15, 8, 5))
    pygame.draw.ellipse(surf, MALTIPOO_NOSE2, (34, 16, 6, 3))
    return surf


# ---------------------------------------------------------------------------
# Bird
# ---------------------------------------------------------------------------
class Bird:
    def __init__(self, sprite_surf):
        self.x     = BIRD_X
        self.y     = SCREEN_H // 2
        self.vel   = 0.0
        self.angle = 0.0
        self._surf = sprite_surf

    def flap(self):
        self.vel = FLAP_STRENGTH

    def update(self):
        self.vel   = min(self.vel + GRAVITY, MAX_FALL)
        self.y    += self.vel
        self.angle = max(-30.0, min(self.vel * 5.0, 90.0))

    def get_rect(self):
        return pygame.Rect(self.x + 4, int(self.y) + 4, BIRD_W - 8, BIRD_H - 8)

    def draw(self, surface):
        rotated = pygame.transform.rotate(self._surf, -self.angle)
        rect    = rotated.get_rect(center=(self.x + BIRD_W // 2,
                                           int(self.y) + BIRD_H // 2))
        surface.blit(rotated, rect)


# ---------------------------------------------------------------------------
# Pipe
# ---------------------------------------------------------------------------
class Pipe:
    _min_top = 120

    def __init__(self, x, gap, speed):
        self.gap     = gap
        self.speed   = speed
        self.x       = x
        max_top      = SCREEN_H - GROUND_H - gap - 60
        self.gap_top = random.randint(self._min_top, max_top)
        self.scored  = False

    def update(self):
        self.x -= self.speed

    def is_off_screen(self):
        return self.x + PIPE_WIDTH < 0

    def get_rects(self):
        top_rect = pygame.Rect(self.x, 0, PIPE_WIDTH, self.gap_top)
        bot_rect = pygame.Rect(self.x, self.gap_top + self.gap,
                               PIPE_WIDTH,
                               SCREEN_H - self.gap_top - self.gap)
        return top_rect, bot_rect

    def draw(self, surface):
        top_rect, bot_rect = self.get_rects()

        # Main pipe bodies
        pygame.draw.rect(surface, PIPE_COLOR, top_rect)
        pygame.draw.rect(surface, PIPE_COLOR, bot_rect)

        # Caps at open ends
        cap_x   = self.x - PIPE_CAP_EXTRA // 2
        cap_w   = PIPE_WIDTH + PIPE_CAP_EXTRA

        # Top pipe bottom cap
        top_cap = pygame.Rect(cap_x, self.gap_top - PIPE_CAP_H, cap_w, PIPE_CAP_H)
        pygame.draw.rect(surface, PIPE_DARK, top_cap)

        # Bottom pipe top cap
        bot_cap = pygame.Rect(cap_x, self.gap_top + self.gap, cap_w, PIPE_CAP_H)
        pygame.draw.rect(surface, PIPE_DARK, bot_cap)


# ---------------------------------------------------------------------------
# Game
# ---------------------------------------------------------------------------
class Game:
    # Select-screen box geometry (shared between draw and event handler)
    _SEL_BOX_W  = 130
    _SEL_BOX_H  = 120
    _SEL_CENTERS = [(110, 235), (290, 235)]

    def __init__(self):
        pygame.init()
        pygame.display.set_caption(TITLE)
        self.screen      = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        self.clock       = pygame.time.Clock()
        self.font_lg     = pygame.font.Font(None, 72)
        self.font_md     = pygame.font.Font(None, 48)
        self.font_sm     = pygame.font.Font(None, 30)
        self.font_xs     = pygame.font.Font(None, 24)
        self.high_score  = 0
        self.selected_char = 0  # 0 = Pillow, 1 = Blanket
        self.char_surfs  = [make_poodle_surface(), make_maltipoo_surface()]
        self.char_names  = ["Pillow", "Blanket"]
        self.state       = "select"
        self._reset()

    def _reset(self):
        self.bird           = Bird(self.char_surfs[self.selected_char])
        self.pipes          = []
        self.score          = 0
        self.frame          = 0
        self.cur_speed      = PIPE_SPEED
        self.cur_gap        = PIPE_GAP
        self.last_diff_score = 0
        self.flash_timer    = 0

    # -- Events --------------------------------------------------------------
    def _handle_events(self):
        bw, bh = self._SEL_BOX_W, self._SEL_BOX_H
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # --- Select screen ---
            if self.state == "select":
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        self.selected_char = 0
                    elif event.key == pygame.K_RIGHT:
                        self.selected_char = 1
                    elif event.key in (pygame.K_SPACE, pygame.K_RETURN):
                        self._reset()
                        self.state = "start"
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mx, my = event.pos
                    for i, (cx, cy) in enumerate(self._SEL_CENTERS):
                        if (cx - bw//2 <= mx <= cx + bw//2 and
                                cy - bh//2 <= my <= cy + bh//2):
                            self.selected_char = i
                            self._reset()
                            self.state = "start"
                            break
                continue

            # --- C key on dead screen → back to select ---
            if (event.type == pygame.KEYDOWN and event.key == pygame.K_c
                    and self.state == "dead"):
                self.state = "select"
                continue

            flap_keys  = event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE
            flap_mouse = event.type == pygame.MOUSEBUTTONDOWN and event.button == 1

            if flap_keys or flap_mouse:
                if self.state == "start":
                    self.state = "playing"
                elif self.state == "playing":
                    self.bird.flap()
                elif self.state == "dead":
                    self.state = "select"

    # -- Update --------------------------------------------------------------
    def _update(self):
        if self.state != "playing":
            return

        self.frame += 1
        self.bird.update()

        # Spawn pipes
        interval = max(1, int(PIPE_SPACING // self.cur_speed))
        if self.frame > 0 and self.frame % interval == 0:
            self.pipes.append(Pipe(SCREEN_W, self.cur_gap, self.cur_speed))

        # Update pipes + score
        for pipe in self.pipes:
            pipe.update()
            if not pipe.scored and pipe.x + PIPE_WIDTH < BIRD_X:
                pipe.scored = True
                self.score += 1
                # Increase difficulty every 5 points
                if self.score % 5 == 0 and self.score != self.last_diff_score:
                    self.last_diff_score = self.score
                    self.cur_speed = min(self.cur_speed + 0.5, 7)
                    self.cur_gap   = max(self.cur_gap   - 10,  100)
                    self.flash_timer = 90

        # Remove off-screen pipes
        self.pipes = [p for p in self.pipes if not p.is_off_screen()]

        # Collision
        if self._check_collision():
            self.high_score = max(self.high_score, self.score)
            self.state = "dead"

    def _check_collision(self):
        bird_rect = self.bird.get_rect()

        # Ground
        if self.bird.y + BIRD_H >= SCREEN_H - GROUND_H:
            return True
        # Ceiling
        if self.bird.y < 0:
            return True
        # Pipes
        for pipe in self.pipes:
            top_r, bot_r = pipe.get_rects()
            if bird_rect.colliderect(top_r) or bird_rect.colliderect(bot_r):
                return True

        return False

    # -- Draw ----------------------------------------------------------------
    def _draw_text(self, text, font, x, y, center=True):
        shadow = font.render(text, True, SHADOW_COLOR)
        main   = font.render(text, True, TEXT_COLOR)
        sr     = shadow.get_rect()
        mr     = main.get_rect()
        if center:
            sr.center = (x + 2, y + 2)
            mr.center = (x,     y)
        else:
            sr.topleft = (x + 2, y + 2)
            mr.topleft = (x,     y)
        self.screen.blit(shadow, sr)
        self.screen.blit(main,   mr)

    def _draw_overlay(self):
        overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        overlay.fill(OVERLAY_COLOR)
        self.screen.blit(overlay, (0, 0))

    def _draw_select_screen(self):
        GOLD = (255, 215, 0)
        DIM  = (90, 90, 90)
        bw, bh = self._SEL_BOX_W, self._SEL_BOX_H

        self.screen.fill(SKY_COLOR)
        self._draw_overlay()

        self._draw_text("CHOOSE YOUR CHARACTER", self.font_md, SCREEN_W // 2, 95)

        for i, (cx, cy) in enumerate(self._SEL_CENTERS):
            bx = cx - bw // 2
            by = cy - bh // 2
            box_rect = pygame.Rect(bx, by, bw, bh)
            border   = GOLD if i == self.selected_char else DIM
            pygame.draw.rect(self.screen, border, box_rect,
                             width=3, border_radius=10)

            # Sprite preview at 2×
            scaled = pygame.transform.scale(
                self.char_surfs[i], (BIRD_W * 2, BIRD_H * 2))
            sr = scaled.get_rect(center=(cx, cy - 8))
            self.screen.blit(scaled, sr)

            self._draw_text(self.char_names[i], self.font_sm, cx, by + bh - 14)

        self._draw_text("Tap a character to play",
                        self.font_sm, SCREEN_W // 2, 390)

    def _draw(self):
        if self.state == "select":
            self._draw_select_screen()
            pygame.display.flip()
            return

        # Background
        self.screen.fill(SKY_COLOR)

        # Pipes
        for pipe in self.pipes:
            pipe.draw(self.screen)

        # Ground
        ground_y = SCREEN_H - GROUND_H
        pygame.draw.rect(self.screen, GROUND_COLOR,
                         pygame.Rect(0, ground_y, SCREEN_W, GROUND_H))
        pygame.draw.rect(self.screen, GROUND_LINE,
                         pygame.Rect(0, ground_y, SCREEN_W, 4))

        # Bird
        self.bird.draw(self.screen)

        # HUD: score during play
        if self.state == "playing":
            self._draw_text(str(self.score), self.font_lg, SCREEN_W // 2, 60)
            level = self.score // 5 + 1
            self._draw_text(f"Lv {level}", self.font_sm, SCREEN_W - 36, 20)
            if self.flash_timer > 0:
                self.flash_timer -= 1

        # --- Start screen ---
        if self.state == "start":
            self._draw_overlay()
            self._draw_text("FLAPPY POODLE", self.font_md, SCREEN_W // 2, SCREEN_H // 2 - 40)
            self._draw_text("Tap to start", self.font_sm, SCREEN_W // 2, SCREEN_H // 2 + 20)

        # --- Game over screen ---
        if self.state == "dead":
            self._draw_overlay()
            self._draw_text("GAME OVER",  self.font_md, SCREEN_W // 2, SCREEN_H // 2 - 70)
            self._draw_text(f"Score: {self.score}",      self.font_sm, SCREEN_W // 2, SCREEN_H // 2)
            self._draw_text(f"Best:  {self.high_score}", self.font_sm, SCREEN_W // 2, SCREEN_H // 2 + 36)
            self._draw_text("Tap to continue",           self.font_sm, SCREEN_W // 2, SCREEN_H // 2 + 96)

        pygame.display.flip()

    # -- Main loop -----------------------------------------------------------
    def run(self):
        while True:
            self._handle_events()
            self._update()
            self._draw()
            self.clock.tick(FPS)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    Game().run()

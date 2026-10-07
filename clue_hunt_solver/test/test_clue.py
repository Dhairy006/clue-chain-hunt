import unittest

from clue_hunt_solver.clue import ClueError, chain_token, parse_clue, validate_clue


class ClueTests(unittest.TestCase):
    def test_start_token(self):
        self.assertEqual(chain_token('START'), '7196')

    def test_all_commands(self):
        self.assertEqual(parse_clue('HUNT:1:7196:GOTO 1.5 -3.9').command, 'GOTO')
        self.assertEqual(parse_clue('HUNT:2:7F49:PILLAR RED').args, ('RED',))
        self.assertEqual(
            parse_clue('HUNT:3:BCB8:BETWEEN BLUE GREEN 0.59').args,
            ('BLUE', 'GREEN', 0.59),
        )
        self.assertEqual(parse_clue('HUNT:4:EC9E:REL 5.08 -0.65').command, 'REL')
        self.assertTrue(
            parse_clue('HUNT:5:1756:TREASURE REL 0.98 3.15').treasure
        )

    def test_practice_chain_and_lookalike(self):
        clues = [
            'HUNT:1:7196:GOTO 1.5 -3.9',
            'HUNT:2:7F49:PILLAR RED',
            'HUNT:3:BCB8:BETWEEN BLUE GREEN 0.59',
            'HUNT:4:EC9E:REL 5.08 -0.65',
            'HUNT:5:1756:TREASURE REL 0.98 3.15',
        ]
        previous = 'START'
        for board_id, text in enumerate(clues, start=1):
            validate_clue(text, board_id, previous)
            previous = text
        with self.assertRaises(ClueError):
            validate_clue('HUNT:4:07A8:REL 2.00 0.00', 4, clues[2])

    def test_rejects_invalid_data(self):
        for text in (
            'HUNT:1:7196:GOTO nan 2',
            'HUNT:1:7196:PILLAR PURPLE',
            'HUNT:1:7196:BETWEEN RED BLUE 1.4',
            'HUNT:1:7196:TREASURE GOTO 1 2',
        ):
            with self.assertRaises(ClueError, msg=text):
                parse_clue(text)


if __name__ == '__main__':
    unittest.main()

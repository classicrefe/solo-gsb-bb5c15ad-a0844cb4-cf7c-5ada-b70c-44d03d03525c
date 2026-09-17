"""Unit tests that run with the Python standard library only.

Run with:  python3 -m unittest discover -s tests -v
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import db as db_mod  # noqa: E402
from app.comparison import compare_configs  # noqa: E402
from app.evaluator import evaluate_flag, hash_bucket  # noqa: E402
from app.validation import ConfigValidationError, validate_config  # noqa: E402


def flag(**kw):
    base = {"enabled": True, "default": False, "prerequisites": [], "rules": []}
    base.update(kw)
    return base


def rule(conditions, value=True, id=None):
    r = {"conditions": conditions, "value": value}
    if id is not None:
        r["id"] = id
    return r


class RuleEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.flags = {
            "beta": flag(
                rules=[
                    rule([{"attr": "tier", "op": "eq", "value": "gold"}], value="gold-banner", id="gold"),
                    rule([{"attr": "tier", "op": "in", "value": ["silver", "bronze"]}], id="tiers"),
                ],
                default="off",
            ),
        }

    def test_first_matching_rule_wins_in_order(self):
        res = evaluate_flag("beta", self.flags, attributes={"tier": "gold"})
        self.assertTrue(res["hit"])
        self.assertEqual(res["value"], "gold-banner")
        self.assertEqual(res["reason"], "rule_match")
        self.assertEqual(res["rule_id"], "gold")

    def test_second_rule_matches(self):
        res = evaluate_flag("beta", self.flags, attributes={"tier": "silver"})
        self.assertEqual(res["rule_id"], "tiers")

    def test_default_when_no_rule_matches(self):
        res = evaluate_flag("beta", self.flags, attributes={"tier": "free"})
        self.assertEqual(res["reason"], "default")
        self.assertEqual(res["value"], "off")
        # "off" is a non-empty string variant; hit is the boolean coercion of
        # the value. Boolean-style flags should use true/false defaults.
        self.assertTrue(res["hit"])

    def test_boolean_default_flag_is_off(self):
        flags = {"f": flag(default=False, rules=[rule([{"attr": "x", "op": "eq", "value": 1}])])}
        res = evaluate_flag("f", flags, attributes={"x": 0})
        self.assertEqual(res["reason"], "default")
        self.assertFalse(res["hit"])

    def test_missing_attribute_is_none(self):
        res = evaluate_flag("beta", self.flags, attributes={})
        self.assertEqual(res["reason"], "default")

    def test_comparison_and_string_ops(self):
        flags = {
            "f": flag(
                rules=[
                    rule([{"attr": "age", "op": "ge", "value": 18}], id="adult"),
                ]
            ),
            "p": flag(
                rules=[rule([{"attr": "email", "op": "suffix", "value": "@acme.com"}], id="acme")]
            ),
        }
        self.assertEqual(evaluate_flag("f", flags, attributes={"age": 21})["rule_id"], "adult")
        self.assertFalse(evaluate_flag("f", flags, attributes={"age": 5})["hit"])
        self.assertTrue(evaluate_flag("p", flags, attributes={"email": "a@acme.com"})["hit"])
        self.assertFalse(evaluate_flag("p", flags, attributes={"email": "a@other.com"})["hit"])

    def test_catch_all_rule_with_empty_conditions(self):
        flags = {"f": flag(rules=[{"conditions": [], "value": True, "id": "all"}])}
        self.assertEqual(evaluate_flag("f", flags, attributes={})["rule_id"], "all")


class DisabledFlagTests(unittest.TestCase):
    def test_disabled_returns_default_even_with_rules(self):
        flags = {"f": flag(enabled=False, default=False, rules=[rule([], id="r")])}
        res = evaluate_flag("f", flags, attributes={"x": 1})
        self.assertEqual(res["reason"], "disabled")
        self.assertFalse(res["hit"])


class PrerequisiteTests(unittest.TestCase):
    def setUp(self):
        self.flags = {
            "logged_in": flag(default=False, rules=[rule([{"attr": "uid", "op": "ne", "value": None}], id="has-uid")]),
            "v2_api": flag(
                prerequisites=[{"flag": "logged_in", "expect": True}],
                default=False,
                rules=[rule([{"attr": "tenant", "op": "eq", "value": "acme"}], id="acme")],
            ),
        }

    def test_chain_when_prerequisite_passes(self):
        res = evaluate_flag("v2_api", self.flags, attributes={"uid": "u-1", "tenant": "acme"})
        self.assertTrue(res["hit"])
        names = [n["flag"] for n in res["chain"]]
        self.assertEqual(names, ["logged_in", "v2_api"])
        self.assertEqual(res["chain"][0]["prerequisite_of"], "v2_api")
        self.assertTrue(res["chain"][0]["expected"])
        self.assertEqual(res["chain"][1]["reason"], "rule_match")

    def test_prerequisite_fails_short_circuits(self):
        res = evaluate_flag("v2_api", self.flags, attributes={"tenant": "acme"})
        self.assertFalse(res["hit"])
        self.assertEqual(res["reason"], "prerequisite_failed")
        # v2's rules must never have been evaluated
        self.assertEqual([n["flag"] for n in res["chain"]], ["logged_in", "v2_api"])
        self.assertEqual(res["chain"][1]["reason"], "prerequisite_failed")

    def test_negated_expect(self):
        flags = {
            "kill": flag(default=False),
            "feature": flag(
                default="on",
                prerequisites=[{"flag": "kill", "expect": False}],
            ),
        }
        res = evaluate_flag("feature", flags, attributes={})
        self.assertEqual(res["value"], "on")

    def test_deep_chain_is_leaf_first(self):
        flags = {
            "a": flag(default=True),
            "b": flag(default=True, prerequisites=[{"flag": "a"}]),
            "c": flag(default="yes", prerequisites=[{"flag": "b"}]),
        }
        res = evaluate_flag("c", flags, attributes={})
        self.assertEqual([n["flag"] for n in res["chain"]], ["a", "b", "c"])
        self.assertEqual(res["value"], "yes")

    def test_transitive_failure_chain_has_no_duplicate_flags(self):
        flags = {
            "a": flag(default=False),
            "b": flag(default=True, prerequisites=[{"flag": "a"}]),
            "c": flag(default="yes", prerequisites=[{"flag": "b"}]),
        }
        res = evaluate_flag("c", flags, attributes={})
        names = [n["flag"] for n in res["chain"]]
        self.assertEqual(names, ["a", "b", "c"])
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(res["reason"], "prerequisite_failed")
        self.assertEqual(res["chain"][0]["prerequisite_of"], "b")
        self.assertEqual(res["chain"][1]["reason"], "prerequisite_failed")
        self.assertEqual(res["chain"][2]["reason"], "prerequisite_failed")

    def test_diamond_dependency_evaluates_shared_flag_once(self):
        flags = {
            "a": flag(rules=[rule([{"attr": "tier", "op": "eq", "value": "gold"}], id="gold")]),
            "b": flag(default=True, prerequisites=[{"flag": "a"}]),
            "c": flag(default="on", prerequisites=[{"flag": "a"}, {"flag": "b"}]),
        }
        res = evaluate_flag("c", flags, attributes={"tier": "gold"})
        names = [n["flag"] for n in res["chain"]]
        self.assertEqual(names, ["a", "b", "c"])
        self.assertEqual(res["value"], "on")


class RolloutTests(unittest.TestCase):
    def test_stable_bucket_for_same_flag_and_user(self):
        b1 = hash_bucket("new_checkout", "user-42")
        b2 = hash_bucket("new_checkout", "user-42")
        self.assertEqual(b1, b2)
        self.assertIsInstance(b1, int)
        self.assertGreaterEqual(b1, 0)
        self.assertLess(b1, 100)

    def test_different_flags_distribute_independently(self):
        # Extremely unlikely that all flags collide for the same user.
        user = "user-42"
        buckets = {hash_bucket(f"flag-{i}", user) for i in range(20)}
        self.assertGreater(len(buckets), 1)

    def test_zero_percentage_never_in(self):
        flags = {"f": flag(rollout={"percentage": 0, "key": "user_id", "value": True})}
        for uid in range(50):
            res = evaluate_flag("f", flags, user_id=str(uid))
            self.assertFalse(res["hit"])
            self.assertEqual(res["reason"], "rollout_out")

    def test_hundred_percentage_always_in(self):
        flags = {"f": flag(rollout={"percentage": 100, "key": "user_id", "value": "x"})}
        for uid in range(50):
            res = evaluate_flag("f", flags, user_id=str(uid))
            self.assertEqual(res["reason"], "rollout_in")
            self.assertEqual(res["value"], "x")

    def test_inclusion_is_monotonic_as_percentage_grows(self):
        flags_cfg = lambda p: {"f": flag(rollout={"percentage": p, "key": "user_id", "value": True})}
        for uid in range(300):
            bucket = hash_bucket("f", str(uid))
            in_at = lambda p: evaluate_flag("f", flags_cfg(p), user_id=str(uid))["hit"]
            if bucket < 10:
                self.assertTrue(in_at(10))
            if not in_at(30):
                self.assertFalse(in_at(10))
            self.assertTrue(in_at(100))

    def test_stability_repeated_evaluation(self):
        flags = {"f": flag(rollout={"percentage": 50, "key": "user_id"})}
        first = evaluate_flag("f", flags, user_id="alice")
        for _ in range(20):
            again = evaluate_flag("f", flags, user_id="alice", attributes={"plan": "pro"})
            self.assertEqual(again["hit"], first["hit"])
            self.assertEqual(again["rollout"]["bucket"], first["rollout"]["bucket"])

    def test_no_user_id_falls_out_of_rollout(self):
        flags = {"f": flag(rollout={"percentage": 100, "key": "user_id", "value": True})}
        res = evaluate_flag("f", flags, attributes={})
        self.assertFalse(res["hit"])
        self.assertEqual(res["reason"], "rollout_out")

    def test_numeric_user_id_is_supported(self):
        flags = {"f": flag(rollout={"percentage": 100, "key": "user_id", "value": True})}
        res = evaluate_flag("f", flags, user_id=12345)
        self.assertTrue(res["hit"])
        self.assertEqual(res["rollout"]["bucket"], hash_bucket("f", "12345"))


class ValidationTests(unittest.TestCase):
    def _expect(self, config, code):
        with self.assertRaises(ConfigValidationError) as ctx:
            validate_config(config)
        codes = [e["code"] for e in ctx.exception.errors]
        self.assertIn(code, codes, msg=f"expected {code} in {codes}")

    def test_cycle_of_two_is_rejected(self):
        config = {
            "flags": {
                "a": {"prerequisites": [{"flag": "b"}], "rules": [], "default": False},
                "b": {"prerequisites": [{"flag": "a"}], "rules": [], "default": False},
            }
        }
        self._expect(config, "circular_dependency")

    def test_self_cycle_is_rejected(self):
        config = {"flags": {"a": {"prerequisites": [{"flag": "a"}]}}}
        self._expect(config, "self_dependency")

    def test_three_node_cycle_is_rejected(self):
        config = {
            "flags": {
                "a": {"prerequisites": [{"flag": "b"}]},
                "b": {"prerequisites": [{"flag": "c"}]},
                "c": {"prerequisites": [{"flag": "a"}]},
            }
        }
        self._expect(config, "circular_dependency")

    def test_dag_is_accepted(self):
        config = {
            "flags": {
                "a": {},
                "b": {"prerequisites": [{"flag": "a"}]},
                "c": {"prerequisites": [{"flag": "a"}, {"flag": "b"}]},
            }
        }
        validate_config(config)  # should not raise

    def test_percentage_bounds(self):
        for bad in (-1, 101):
            self._expect(
                {"flags": {"f": {"rollout": {"percentage": bad}}}},
                "invalid_percentage",
            )

    def test_percentage_must_be_integer(self):
        self._expect(
            {"flags": {"f": {"rollout": {"percentage": "50"}}}},
            "invalid_percentage",
        )
        # bool must not be accepted as int
        self._expect(
            {"flags": {"f": {"rollout": {"percentage": True}}}},
            "invalid_percentage",
        )

    def test_boundary_percentages_accepted(self):
        for good in (0, 50, 100):
            validate_config({"flags": {"f": {"rollout": {"percentage": good}}}})

    def test_unknown_operator_rejected(self):
        self._expect(
            {"flags": {"f": {"rules": [{"conditions": [{"attr": "x", "op": "regex"}]}]}}},
            "unknown_operator",
        )

    def test_missing_prerequisite_rejected(self):
        self._expect(
            {"flags": {"f": {"prerequisites": [{"flag": "ghost"}]}}},
            "missing_prerequisite",
        )

    def test_duplicate_rule_id_rejected(self):
        config = {
            "flags": {
                "f": {
                    "rules": [
                        {"id": "r1", "conditions": []},
                        {"id": "r1", "conditions": [{"attr": "a", "op": "eq", "value": 1}]},
                    ]
                }
            }
        }
        self._expect(config, "duplicate_rule_id")

    def test_unsupported_rollout_key_rejected(self):
        self._expect(
            {"flags": {"f": {"rollout": {"percentage": 10, "key": "email"}}}},
            "invalid_rollout_key",
        )

    def test_empty_config_is_valid(self):
        validate_config({"flags": {}})


class ImmutableVersionStorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["FLAG_DB_PATH"] = os.path.join(self.tmp.name, "test.db")
        self.conn = db_mod.connect()
        db_mod.init_db(self.conn)

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()
        os.environ.pop("FLAG_DB_PATH", None)

    def test_versions_are_immutable_and_history_is_preserved(self):
        v1 = {"flags": {"f": {"enabled": True, "default": "v1", "rules": []}}}
        v2 = {"flags": {"f": {"enabled": True, "default": "v2", "rules": []}}}

        ts1 = db_mod.insert_version(self.conn, "v1", v1)
        self.assertIsNotNone(ts1)
        self.assertIsNone(db_mod.insert_version(self.conn, "v1", v2))  # duplicate rejected

        loaded_v1 = db_mod.get_version(self.conn, "v1")
        self.assertEqual(loaded_v1["config"]["flags"]["f"]["default"], "v1")

        db_mod.insert_version(self.conn, "v2", v2)
        self.assertEqual(db_mod.get_version(self.conn, "v2")["config"]["flags"]["f"]["default"], "v2")

        # Publishing v2 must not change evaluations pinned to v1.
        still_v1 = db_mod.get_version(self.conn, "v1")["config"]
        res = evaluate_flag("f", still_v1["flags"], attributes={})
        self.assertEqual(res["value"], "v1")

        versions = db_mod.list_versions(self.conn)
        self.assertEqual([v["version"] for v in versions], ["v1", "v2"])
        self.assertEqual(versions[0]["flag_count"], 1)


class ConfigComparisonTests(unittest.TestCase):
    def setUp(self):
        # v1 dependency DAG (edge = prerequisite):
        #   a -> b ; a -> c ; b -> d ; c -> d ; d -> e
        self.v1 = {
            "flags": {
                "a": flag(),
                "b": flag(prerequisites=[{"flag": "a"}]),
                "c": flag(prerequisites=[{"flag": "a"}]),
                "d": flag(prerequisites=[{"flag": "b"}, {"flag": "c"}]),
                "e": flag(prerequisites=[{"flag": "d"}]),
                "p": flag(),
            }
        }

    def test_added_removed_modified_are_classified_and_sorted(self):
        v2 = {
            "flags": {
                # unchanged
                "b": flag(prerequisites=[{"flag": "a"}]),
                # c removed; a and e modified
                "a": flag(default=True),
                "d": flag(prerequisites=[{"flag": "b"}, {"flag": "c"}]),
                "e": flag(enabled=False),
                # new flag that nothing depends on yet
                "p": flag(),
                "z": flag(),
            }
        }
        result = compare_configs(self.v1, v2)
        self.assertEqual(result.added, ["z"])
        self.assertEqual(result.removed, ["c"])
        self.assertEqual(result.modified, ["a", "e"])
        self.assertEqual(list(result.impacts), ["a", "c", "e", "z"])

    def test_shortest_path_wins_and_ties_are_lexicographic(self):
        v2 = {
            "flags": {
                "a": flag(default=True),  # modified
                "b": flag(prerequisites=[{"flag": "a"}]),
                "c": flag(prerequisites=[{"flag": "a"}]),
                "d": flag(prerequisites=[{"flag": "b"}, {"flag": "c"}]),
                "e": flag(prerequisites=[{"flag": "d"}]),
                "p": flag(),
            }
        }
        impacted = {item["flag"]: item["path"] for item in compare_configs(self.v1, v2).impacts["a"]}
        self.assertEqual(
            impacted,
            {
                "b": ["a", "b"],
                "c": ["a", "c"],
                # two equal-length paths a->b->d and a->c->d: take the lexicographically smaller
                "d": ["a", "b", "d"],
                "e": ["a", "b", "d", "e"],
            },
        )

    def test_indirect_impact_is_deduplicated_per_changed_flag(self):
        v2 = {
            "flags": {
                "a": flag(default=True),
                "b": flag(prerequisites=[{"flag": "a"}]),
                "c": flag(prerequisites=[{"flag": "a"}]),
                "d": flag(prerequisites=[{"flag": "b"}, {"flag": "c"}]),
                "e": flag(prerequisites=[{"flag": "d"}]),
                "p": flag(),
            }
        }
        affected = [item["flag"] for item in compare_configs(self.v1, v2).impacts["a"]]
        self.assertEqual(affected, sorted(affected))
        self.assertEqual(len(affected), len(set(affected)))

    def test_removed_flag_propagates_over_union_of_both_versions_edges(self):
        v2 = {
            "flags": {
                # a is removed; b/c/d/e survive (the published v2 has its
                # own graph, but the union must still propagate via v1 edges)
                "b": flag(),
                "c": flag(),
                "d": flag(prerequisites=[{"flag": "b"}, {"flag": "c"}]),
                "e": flag(prerequisites=[{"flag": "d"}]),
                "p": flag(),
            }
        }
        impacted = {item["flag"]: item["path"] for item in compare_configs(self.v1, v2).impacts["a"]}
        self.assertEqual(impacted["b"], ["a", "b"])
        self.assertEqual(impacted["e"], ["a", "b", "d", "e"])

    def test_new_edge_in_to_version_is_part_of_the_union(self):
        # 'a' is modified and gains a new dependent 'p' only present via an
        # edge of the to-version; the union graph must still reach it.
        v2 = {
            "flags": {
                "a": flag(default=True),
                "b": flag(prerequisites=[{"flag": "a"}]),
                "c": flag(prerequisites=[{"flag": "a"}]),
                "d": flag(prerequisites=[{"flag": "b"}, {"flag": "c"}]),
                "e": flag(prerequisites=[{"flag": "d"}]),
                "p": flag(prerequisites=[{"flag": "a"}]),
            }
        }
        impacted_a = {
            item["flag"]: item["path"]
            for item in compare_configs(self.v1, v2).impacts["a"]
        }
        self.assertEqual(impacted_a["p"], ["a", "p"])
        self.assertEqual(impacted_a["d"], ["a", "b", "d"])

    def test_added_and_leaf_flags_have_no_impact(self):
        v2 = {
            "flags": {
                "a": flag(),
                "b": flag(prerequisites=[{"flag": "a"}]),
                "c": flag(prerequisites=[{"flag": "a"}]),
                "d": flag(prerequisites=[{"flag": "b"}, {"flag": "c"}]),
                "e": flag(prerequisites=[{"flag": "d"}]),
                "p": flag(default=True),  # modified leaf
                "q": flag(),  # added leaf
            }
        }
        impacts = compare_configs(self.v1, v2).impacts
        self.assertEqual(impacts["p"], [])
        self.assertEqual(impacts["q"], [])

    def test_key_reordering_is_not_a_modification(self):
        v1 = {"flags": {"f": {"enabled": True, "default": False, "rules": []}}}
        v2 = {"flags": {"f": {"rules": [], "default": False, "enabled": True}}}
        result = compare_configs(v1, v2)
        self.assertEqual(result.added, [])
        self.assertEqual(result.removed, [])
        self.assertEqual(result.modified, [])
        self.assertEqual(result.impacts, {})

    def test_identical_configs_have_no_changes(self):
        result = compare_configs(self.v1, self.v1)
        self.assertEqual(result.added, [])
        self.assertEqual(result.removed, [])
        self.assertEqual(result.modified, [])
        self.assertEqual(result.impacts, {})

    def test_any_definition_delta_is_modified(self):
        v2 = {
            "flags": {
                "a": flag(rules=[{"conditions": [], "value": True}]),
                "b": flag(prerequisites=[{"flag": "a"}]),
                "c": flag(prerequisites=[{"flag": "a"}]),
                "d": flag(prerequisites=[{"flag": "b"}, {"flag": "c"}]),
                "e": flag(prerequisites=[{"flag": "d"}]),
                "p": flag(),
            }
        }
        result = compare_configs(self.v1, v2)
        self.assertEqual(result.modified, ["a"])


class EndToEndPinnedVersionTests(unittest.TestCase):
    def test_pinned_version_result_is_stable(self):
        flags_v1 = {
            "checkout": flag(
                rollout={"percentage": 50, "key": "user_id"},
                prerequisites=[{"flag": "signed_in"}],
            ),
            "signed_in": flag(rules=[rule([{"attr": "user_id", "op": "ne", "value": None}])]),
        }
        for uid in range(100):
            r1 = evaluate_flag("checkout", flags_v1, user_id=str(uid))
            r2 = evaluate_flag("checkout", flags_v1, user_id=str(uid), attributes={"x": "y"})
            self.assertEqual(r1["hit"], r2["hit"])
            self.assertEqual([n["flag"] for n in r1["chain"]], ["signed_in", "checkout"])


if __name__ == "__main__":
    unittest.main()

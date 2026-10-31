from pyrulearn.data import DataSpecBuilder, merge_dataspecs


def test_merge_numeric_union():
    a = DataSpecBuilder()
    a.add_numeric("age", [20, 30])
    ds_a = a.build()

    b = DataSpecBuilder()
    b.add_numeric("age", [25, 30, 40])
    ds_b = b.build()

    merged = merge_dataspecs(ds_a, ds_b).build()
    # negation is the builder default now, so each ge feature is paired
    # with its exact "<" negation feature -- merge must carry both through
    assert set(merged.feature_names) == {
        "age>=20", "age>=25", "age>=30", "age>=40",
        "age<20", "age<25", "age<30", "age<40",
    }


def test_merge_nominal_union():
    a = DataSpecBuilder()
    a.add_nominal("color", ["red", "green"])
    ds_a = a.build()

    b = DataSpecBuilder()
    b.add_nominal("color", ["green", "blue"])
    ds_b = b.build()

    merged = merge_dataspecs(ds_a, ds_b).build()
    assert set(merged.feature_names) == {
        "color=red", "color=green", "color=blue",
        "color!=red", "color!=green", "color!=blue",
    }


def test_merge_one_sided_attribute():
    a = DataSpecBuilder()
    a.add_boolean("smoker")
    a.add_numeric("age", [30])
    ds_a = a.build()

    b = DataSpecBuilder()
    b.add_boolean("smoker")
    ds_b = b.build()

    merged = merge_dataspecs(ds_a, ds_b).build()
    assert set(merged.feature_names) == {"smoker", "not smoker", "age>=30", "age<30"}


def test_merge_boolean():
    a = DataSpecBuilder()
    a.add_boolean("smoker")
    ds_a = a.build()

    b = DataSpecBuilder()
    b.add_boolean("smoker")
    ds_b = b.build()

    merged = merge_dataspecs(ds_a, ds_b).build()
    assert merged.feature_names == ["smoker", "not smoker"]


def test_merge_type_conflict_raises():
    a = DataSpecBuilder()
    a.add_numeric("code", [1, 2])
    ds_a = a.build()

    b = DataSpecBuilder()
    b.add_nominal("code", ["x", "y"])
    ds_b = b.build()

    try:
        merge_dataspecs(ds_a, ds_b)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_merge_hierarchical_identical_ok():
    tree = {"Europe": {"France": {}, "Germany": {}}, "Asia": {}}
    a = DataSpecBuilder()
    a.add_hierarchical("region", tree)
    ds_a = a.build()

    b = DataSpecBuilder()
    b.add_hierarchical("region", tree)
    ds_b = b.build()

    merged = merge_dataspecs(ds_a, ds_b).build()
    assert set(merged.feature_names) == {
        "region=Europe", "region=France", "region=Germany", "region=Asia",
    }


def test_merge_hierarchical_mismatch_raises():
    a = DataSpecBuilder()
    a.add_hierarchical("region", {"Europe": {"France": {}}})
    ds_a = a.build()

    b = DataSpecBuilder()
    b.add_hierarchical("region", {"Europe": {"Germany": {}}})
    ds_b = b.build()

    try:
        merge_dataspecs(ds_a, ds_b)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_merge_carries_over_missing_name_and_values():
    a = DataSpecBuilder()
    a.add_nominal("color", ["red", "green"], missing_name="<missing>", missing_values=("?",))
    ds_a = a.build()

    b = DataSpecBuilder()
    b.add_nominal("color", ["green", "blue"])
    ds_b = b.build()

    merged = merge_dataspecs(ds_a, ds_b).build()
    assert "color=<missing>" in merged.feature_names
    assert merged.attributes["color"].missing_name == "<missing>"
    assert merged.attributes["color"].missing_values == ("?",)


def test_merge_conflicting_missing_name_raises():
    a = DataSpecBuilder()
    a.add_nominal("color", ["red"], missing_name="<missing>")
    ds_a = a.build()

    b = DataSpecBuilder()
    b.add_nominal("color", ["blue"], missing_name="unknown")
    ds_b = b.build()

    try:
        merge_dataspecs(ds_a, ds_b)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_merge_relational_identical_ok_and_mismatch_raises():
    a = DataSpecBuilder()
    a.add_numeric("age", [30])
    a.add_numeric("income", [50000])
    a.add_relational("rich_for_age", ["income", "age"], expression="income > age * 1000")
    ds_a = a.build()

    b = DataSpecBuilder()
    b.add_numeric("age", [40])
    b.add_numeric("income", [60000])
    b.add_relational("rich_for_age", ["income", "age"], expression="income > age * 1000")
    ds_b = b.build()

    merged = merge_dataspecs(ds_a, ds_b).build()
    assert "rich_for_age" in merged.feature_names
    assert merged.feature_names.count("age>=30") == 1
    assert "age>=40" in merged.feature_names

    c = DataSpecBuilder()
    c.add_numeric("age", [30])
    c.add_numeric("income", [50000])
    c.add_relational("rich_for_age", ["income", "age"], expression="income > age * 2000")  # different expression
    ds_c = c.build()

    try:
        merge_dataspecs(ds_a, ds_c)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_merge_binary_attributes():
    from pyrulearn.data.attributes import AttributeType

    def spec(kind, values):
        b = DataSpecBuilder()
        (b.add_binary if kind == "binary" else b.add_nominal)("x", values)
        return b.build()

    same = merge_dataspecs(spec("binary", ["a", "b"]), spec("binary", ["b", "a"])).build()
    assert same.attributes["x"].type == AttributeType.BINARY
    assert same.feature_names == ["x=a", "x=b"]
    # a wider union is no longer a closed two-value set
    for other in (spec("nominal", ["a", "b", "c"]), spec("binary", ["a", "c"])):
        merged = merge_dataspecs(spec("binary", ["a", "b"]), other).build()
        assert merged.attributes["x"].type == AttributeType.NOMINAL
        assert set(merged.attributes["x"].domain) == {"a", "b", "c"}
        assert "x!=a" in merged.feature_names


def test_remap_inequality_onto_binary_becomes_the_other_value():
    from pyrulearn.rule import Rule
    b = DataSpecBuilder()
    b.add_nominal("x", ["a", "b"])
    nominal = b.build()
    rule = Rule.from_pos_neg(pos=[], neg=[nominal.feature_index("x=a")], target="t", dataspec=nominal)
    assert [nominal.feature_name(l.feature) for l in rule.conditions] == ["x!=a"]
    b = DataSpecBuilder()
    b.add_binary("x", ["a", "b"])
    binary = b.build()
    remapped = rule.remap(binary)
    assert [binary.feature_name(l.feature) for l in remapped.conditions] == ["x=b"]
    # onto a wider nominal, x!=a stays x!=a
    b = DataSpecBuilder()
    b.add_nominal("x", ["a", "b", "c"])
    wide = b.build()
    assert [wide.feature_name(l.feature) for l in rule.remap(wide).conditions] == ["x!=a"]

# Included by runtests.jl; also usable for a focused precision regression.
function test_reachability_precision(case, device)
    bats = sort(collect(case.batteries); by = b -> b.index)
    dt = stage_hours(case)
    _, move, _ = configure_device(device)
    expected = (energy_min = [b.energy_min for b in bats],
                energy_max = [b.energy_max for b in bats],
                charge_gain = [b.charge_efficiency * dt * b.charge_max for b in bats],
                discharge_drop = [(dt / b.discharge_efficiency) * b.discharge_max for b in bats],
                alpha = [b.self_discharge for b in bats])
    @testset "physical reachability precision" begin
        for arch in BATTERY_ARCHITECTURES
            p = move(battery_reachable_policy(case, [4]; n_observation = 2,
                                              head_layers = [4], architecture = arch))
            for key in keys(expected)
                @test eltype(getproperty(p, key)) === Float64
                @test Array(getproperty(p, key)) == getproperty(expected, key)
            end
            @test Array(move([1.00000000000001])) == [1.00000000000001]
            e = move(initial_energy(case))
            @test eltype(e) === Float64
            for t in 1:24
                # Exercise both physical endpoints independently of the learned
                # head, including a Float32 activation with Float64 state.
                y = move(fill(isodd(t) ? 0.0f0 : 1.0f0, length(bats)))
                p.architecture = :tsddr_nonlinear
                target = feasible_target(p, y, e)
                @test eltype(target) === Float64
                eh, th = Array(e), Array(target)
                for (j, b) in enumerate(bats)
                    lo, hi = reachable_interval(b, eh[j], dt)
                    # Separate CPU/GPU multiply-add ordering can differ by an
                    # ulp. This bound scales with state arithmetic precision.
                    tol = 8eps(Float64) * max(1.0, abs(lo), abs(hi))
                    @test lo - tol <= th[j] <= hi + tol
                end
                e = target
            end
            p.architecture = arch
            # Old Flux.state payloads contain metadata. Loading must leave the
            # freshly constructed case constants untouched even if it is bad.
            poisoned = deepcopy(Flux.state(Flux.cpu(p)))
            poisoned.energy_min .= -123.0
            poisoned.alpha .= 0.5
            before = map(a -> copy(Array(a)), Flux.trainables(p))
            load_stateconditioned_policy!(p, poisoned)
            @test map(Array, Flux.trainables(p)) == before
            @test Array(p.energy_min) == expected.energy_min
            @test Array(p.alpha) == expected.alpha
            mktempdir() do dir
                f = joinpath(dir, "policy.jld2")
                save_checkpoint(f, p, Dict{String,Any}(); opt_state = Optimisers.setup(Optimisers.Adam(1e-4), p))
                payload = JLD2.load(f)
                @test keys(payload["state"]) == (:encoder, :combiner)
                load_checkpoint!(p, f)
                @test map(Array, Flux.trainables(p)) == before
                @test Array(p.alpha) == expected.alpha
                legacy = joinpath(dir, "legacy.jld2")
                legacy_meta = copy(payload["meta"])
                delete!(legacy_meta, "reachability_arithmetic")
                # A legacy parent carrying a selection, as every segment
                # checkpoint in the study does.
                legacy_meta["best_cost"] = 123.0
                legacy_meta["best_index"] = 7
                JLD2.jldsave(legacy; state = poisoned, meta = legacy_meta,
                             opt_state = payload["opt_state"])
                @test_throws ErrorException load_checkpoint!(p, legacy; case = case)
                loaded = load_checkpoint!(p, legacy; case = case, allow_precision_upgrade = true)
                @test loaded.meta["reachability_precision_upgraded"]
                @test Array(p.alpha) == expected.alpha
                @test map(Array, Flux.trainables(p)) == before
                # An upgraded parent may not hand its old score to the new run:
                # that number was measured under the other forward map, and
                # nothing in this run would ever re-observe it.
                @test resumed_best_score(loaded.meta["best_cost"],
                                         loaded.meta["best_index"],
                                         loaded.meta["reachability_precision_upgraded"]) ==
                      (Inf, 0)
                # An identical continuation keeps its incumbent, or a resume
                # would re-select a checkpoint it had already rejected.
                @test resumed_best_score(123.0, 7, false) == (123.0, 7)
                # The historical value stays readable on the parent, so dropping
                # the incumbent loses no evidence.
                @test loaded.meta["best_cost"] == 123.0
                @test loaded.meta["best_index"] == 7
            end
        end
        # Differentiate the state path itself, at an interior smooth point.
        p = move(Flux.f64(battery_reachable_policy(case, [4]; n_observation = 2)))
        e = move(initial_energy(case))
        y = move(fill(0.43, length(bats)))
        f(x) = sum(feasible_target(p, y, x))
        g = only(Zygote.gradient(f, e))
        d = move(fill(1.0, length(bats)))
        h = 1e-6
        fd = (f(e .+ h .* d) - f(e .- h .* d)) / (2h)
        @test isapprox(sum(g .* d), fd; rtol = 1e-6, atol = 1e-8)
    end
end

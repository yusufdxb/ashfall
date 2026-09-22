# Research history index

The five studies were developed on research branches that are kept in a private archive and are not
published. That history also contained internal planning documents; publishing it would publish them. This
public release therefore starts from the previously public history (`37b61f1`) and adds the final archive
content as new commits. The file contents match the private archive except as listed in
[`PUBLIC_ARTIFACT_MANIFEST.md`](PUBLIC_ARTIFACT_MANIFEST.md) and [`ARCHIVE.md`](ARCHIVE.md#public-release).

Result records, preregistrations and this repository's documents cite commits from that private history
(for example, a preregistration "committed with its code at `3205f02`"). Those commits are not reachable
here. This index lists every one of them, in order, so each citation can be matched to a full hash, a date
and a subject. What can be checked publicly is the bytes of the records and artifacts (SHA-256 pinned by
the test suite); the commit order below is attested by the private archive, not verifiable from this
repository alone.

Private archive: branch `archive/ashfall-negative-results-2026`, final commit
`dfb9c66a071b42c889842542b3ef9297be526ce8`, annotated tag `archive/ashfall-2026`. Its first commit's parent is
`37b61f1`, the public base of this release.

| # | Commit | Date | Subject |
|---|---|---|---|
| 1 | `f3390872990d913a412a7fd5d3f448b93bfdf0e8` | 2026-09-21 | feat(fbr): failure capsule, boundary replay sampler, acceptance checks, slip cart-pole study |
| 2 | `16011b0ddfa53dc5535a091f7e903a048a3aee6d` | 2026-09-21 | evidence(fbr-toy): 12-seed slip cart-pole study, primary null, plus exploratory budget-curve mode |
| 3 | `896657a889a9c1cca4103bc5137ac99300425d8f` | 2026-09-21 | fix(fbr-toy): register the --efficiency-from flag the efficiency mode reads |
| 4 | `3c5d38aa47200d37b7ef06a103f1ae6432deb29a` | 2026-09-21 | evidence(fbr-toy): exploratory budget curves; no budget at which FBR leads broad DR |
| 5 | `f3ccc1b9786c39eecadb1c948ca2bddb1d19527b` | 2026-09-21 | fix(fbr): carry the boundary censoring kind in a typed exception |
| 6 | `8f49f025a7abe7acb13ecfa421ed7a041342b039` | 2026-09-22 | prereg(fbr-toy-v2): decision gate before GO2 compute, registered with its code |
| 7 | `08adcff3d62dd1bf76468b53609cd6cbce9cfc77` | 2026-09-22 | evidence(fbr-toy-v2): preregistered gate returns NO-GO |
| 8 | `ad463df963111f695a80e91a9ca3744b8db30750` | 2026-09-22 | docs: rebuild the Ashfall story around Failure-Boundary Replay |
| 9 | `48c8c713be8da438eec866e2d16e00ba3d341952` | 2026-09-22 | evidence(fbr): freeze the negative FBR toy result and commit the v2 artifacts |
| 10 | `3205f02fba84d23f06481813afc23d6328dd293e` | 2026-09-22 | prereg(precursor-sweep): replay start time along a failed trajectory, registered with its code |
| 11 | `dfec13088a286bc14499c57a70d435338c12e5e2` | 2026-09-22 | evidence(precursor-sweep): start time matters, the real failure does not; NO-GO |
| 12 | `3ed7f085a1d6f16ea2a5ef732d5adacff840149d` | 2026-09-22 | docs(transition): record H1 and H2 as rejected and state recoverability as a new, unregistered hypothesis |
| 13 | `5c744a8d23c48a3fbc603eb2caf496369ea9010e` | 2026-09-22 | docs(related-work): novelty gate for recoverability-based replay selection |
| 14 | `2227de140e1017b1b83795a178fc9a0e6e587170` | 2026-09-22 | feat(recoverability): branched-continuation estimator, leak-free dataset, analysis plan frozen before estimation |
| 15 | `61bb95d36a6592adcb6039f6169e941ff7bf1d41` | 2026-09-22 | evidence(recoverability): inverted U exists, but timing explains repair value better; NO-GO |
| 16 | `d1f9663cc8753f3264504ae6f224517587ffa549` | 2026-09-22 | docs(transition): H1-H3 stay rejected; failure-critical simulator diagnosis stated as a new hypothesis |
| 17 | `13072552a5f96112307f2f5966250d8f0bb0de52` | 2026-09-22 | docs(related-work): FCSI novelty audit; Gate 1 passes only as a combination claim |
| 18 | `1b657c1fac3da08c689fe9023fb5e5306266a05c` | 2026-09-22 | prereg(fcsi-toy): failure-critical sysid ground-truth toy, registered with its code |
| 19 | `d49daeb3c060c7af3f9cb866b81a7c56b94f239c` | 2026-09-22 | evidence(fcsi-toy): failure-event-conditioned sysid loses to multiple shooting; NO-GO |
| 20 | `feac54a0b18fc85fa32b6551e1cddc5eba8e8189` | 2026-09-22 | docs(transition): four NO-GOs stay frozen; active simulator diagnosis stated as a new hypothesis |
| 21 | `03e9936fe816c28780e0e692977a47fad2e189b4` | 2026-09-22 | docs(related-work): active diagnosis audit; closed-set safe discrimination is prior art (Ni et al. 2026) |
| 22 | `ad32ac0765838c91483caf1d5d2c4ea5e4ad808a` | 2026-09-22 | feat(active-diag): passive weights, safe probe space, EIG probe search, UNKNOWN test, sequential loop |
| 23 | `513be90742ed73e8137de747ee289ca1b0ea9354` | 2026-09-22 | fix(active-diag): development corrections before preregistration |
| 24 | `2b3e40e6b2664ce2e4a4e980d969681c1ba16812` | 2026-09-22 | fix(active-diag): UNKNOWN test uses the best-fitting parameter at fine resolution |
| 25 | `b86006de8f6945b276ee1de6d948deb8dfb499f6` | 2026-09-22 | prereg(active-diag): open-set active diagnosis toy, frozen with code 2b3e40e and development calibration |
| 26 | `1ea85eb6154d1abb5fc4f3ef4b08563cde0c14a2` | 2026-09-22 | fix(active-diag): safety set always contains the MAP value when parameter weights tie |
| 27 | `4186bbf098beb59b0bdc3764b546f971c5bd0047` | 2026-09-22 | evidence(active-diag): chosen probes no better than fixed or random; library-certified probes unsafe under unknown dynamics; NO-GO |
| 28 | `ca6e05551863b16e46b814581538600218c54acc` | 2026-09-22 | docs: mark Ashfall research program archived |
| 29 | `84ac937cc8221ad1c2ee7f1af00543be9df08305` | 2026-09-22 | docs: rewrite README around negative results |
| 30 | `7bd6d6ce7d0518a04da482a31953e7697e1bc7a3` | 2026-09-22 | docs: consolidate archival evidence and lessons |
| 31 | `93c435f0517d232b84fe6edcb0ad4cb271beadb6` | 2026-09-22 | chore: archive superseded documentation |
| 32 | `dfb9c66a071b42c889842542b3ef9297be526ce8` | 2026-09-22 | test: verify frozen research artifacts |

Commits 8 (`ad463df`) and 31 (`93c435f`) also added and later removed three FBR-phase planning documents
(a paper outline, a hardware demo plan and a note on paper structure). They held plans, not evidence, and
are not part of this release.

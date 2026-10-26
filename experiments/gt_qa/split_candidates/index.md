# SPLIT candidates — `dense` GT

105 joins where one GT identity may span two vehicles.
Sorted worst first. Every row is a *question*, not a verdict —
a motorcycle emerging from behind a bus really does reappear
displaced and rescaled.

## How to read a sheet

- **Top row** — full frame either side of the gap, this identity in
  yellow. Look at where on the road the box sits.
- **Middle row** — zoom on the two boxes. Same vehicle or not?
- **Bottom strip** — box diagonal (yellow) and centre-y (blue) over
  the whole track, red line at the gap. Traffic flows toward the
  camera, so one vehicle grows monotonically and drifts down the
  frame. A diagonal that shrinks then regrows is two vehicles.

## The `gt.txt` id is not the CVAT id

CVAT renumbers every track `1..N` in export order when it writes
MOT 1.1 — the pre-labels went in with tracker ids (`5849`, `6413`,
…) and came back as `1..335` with no gaps. So the `gt id` column
below exists **only inside `gt.txt`**; typing it into CVAT finds
nothing or, worse, finds the wrong object. It is kept here so the
rows can be traced back to the sensitivity analysis, not for
lookup.

**Look things up by frame and position instead:**

1. Type the *after* frame number into CVAT's frame box.
2. Find the box inside the red ring on the sheet — the centre
   pixel is printed under each panel (images are 800×450, same
   coordinates CVAT uses).
3. Click it. Whatever id CVAT shows is the real one.
4. Confirm by stepping back to the *before* frame: if it is the
   same CVAT id sitting on a different vehicle, this row is real.
5. Then use *Split* in the object sidebar at the after frame.

Frame numbering: CVAT = MOT − 1 · absolute = MOT − 1 + 10700

| # | gt.txt id | class | CVAT before → after | gap | move | area | centre after | sheet |
|---|---|---|---|---|---|---|---|---|
| 1 | 107 | motorcycle | f247 → f272 | 25f | 11.1 | 13.7x | (650, 344) | [png](001_id107_f247-272.png) |
| 2 | 107 | motorcycle | f226 → f247 | 21f | 4.6 | 11.2x | (453, 195) | [png](002_id107_f226-247.png) |
| 3 | 37 | motorcycle | f111 → f135 | 24f | 4.7 | 9.7x | (355, 174) | [png](003_id37_f111-135.png) |
| 4 | 225 | motorcycle | f378 → f392 | 14f | 9.3 | 2.7x | (370, 186) | [png](004_id225_f378-392.png) |
| 5 | 105 | motorcycle | f241 → f259 | 18f | 9.1 | 1.2x | (141, 138) | [png](005_id105_f241-259.png) |
| 6 | 82 | motorcycle | f271 → f282 | 11f | 3.3 | 8.5x | (291, 174) | [png](006_id82_f271-282.png) |
| 7 | 35 | motorcycle | f18 → f48 | 30f | 7.6 | 7.2x | (486, 191) | [png](007_id35_f18-48.png) |
| 8 | 276 | motorcycle | f537 → f556 | 19f | 1.8 | 7.1x | (261, 129) | [png](008_id276_f537-556.png) |
| 9 | 208 | motorcycle | f329 → f335 | 6f | 1.8 | 7.0x | (621, 404) | [png](009_id208_f329-335.png) |
| 10 | 85 | motorcycle | f134 → f159 | 25f | 6.9 | 2.2x | (327, 143) | [png](010_id85_f134-159.png) |
| 11 | 96 | motorcycle | f276 → f300 | 24f | 1.4 | 6.1x | (587, 307) | [png](011_id96_f276-300.png) |
| 12 | 85 | motorcycle | f121 → f133 | 12f | 2.4 | 6.0x | (397, 117) | [png](012_id85_f121-133.png) |
| 13 | 180 | motorcycle | f296 → f324 | 28f | 6.0 | 1.4x | (493, 251) | [png](013_id180_f296-324.png) |
| 14 | 22 | motorcycle | f127 → f151 | 24f | 5.8 | 1.1x | (788, 418) | [png](014_id22_f127-151.png) |
| 15 | 23 | motorcycle | f99 → f122 | 23f | 5.7 | 1.0x | (789, 412) | [png](015_id23_f99-122.png) |
| 16 | 174 | motorcycle | f375 → f394 | 19f | 5.3 | 2.6x | (728, 365) | [png](016_id174_f375-394.png) |
| 17 | 25 | motorcycle | f47 → f73 | 26f | 5.1 | 4.6x | (376, 118) | [png](017_id25_f47-73.png) |
| 18 | 49 | motorcycle | f91 → f108 | 17f | 5.0 | 1.9x | (511, 265) | [png](018_id49_f91-108.png) |
| 19 | 102 | motorcycle | f215 → f231 | 16f | 2.7 | 4.9x | (792, 434) | [png](019_id102_f215-231.png) |
| 20 | 136 | motorcycle | f258 → f286 | 28f | 4.7 | 1.2x | (661, 430) | [png](020_id136_f258-286.png) |
| 21 | 105 | motorcycle | f191 → f219 | 28f | 4.6 | 1.4x | (505, 329) | [png](021_id105_f191-219.png) |
| 22 | 107 | motorcycle | f312 → f337 | 25f | 4.5 | 1.9x | (646, 416) | [png](022_id107_f312-337.png) |
| 23 | 107 | motorcycle | f298 → f311 | 13f | 4.3 | 2.9x | (365, 430) | [png](023_id107_f298-311.png) |
| 24 | 215 | motorcycle | f407 → f423 | 16f | 4.3 | 1.3x | (788, 424) | [png](024_id215_f407-423.png) |
| 25 | 247 | motorcycle | f481 → f498 | 17f | 2.5 | 4.3x | (417, 402) | [png](025_id247_f481-498.png) |
| 26 | 183 | motorcycle | f360 → f387 | 27f | 4.2 | 1.3x | (642, 312) | [png](026_id183_f360-387.png) |
| 27 | 55 | motorcycle | f70 → f89 | 19f | 4.2 | 2.3x | (190, 158) | [png](027_id55_f70-89.png) |
| 28 | 107 | motorcycle | f272 → f298 | 26f | 3.0 | 4.2x | (434, 268) | [png](028_id107_f272-298.png) |
| 29 | 116 | motorcycle | f232 → f246 | 14f | 2.3 | 4.1x | (785, 439) | [png](029_id116_f232-246.png) |
| 30 | 183 | motorcycle | f434 → f455 | 21f | 4.0 | 2.4x | (727, 352) | [png](030_id183_f434-455.png) |
| 31 | 248 | motorcycle | f483 → f507 | 24f | 4.0 | 1.6x | (743, 441) | [png](031_id248_f483-507.png) |
| 32 | 105 | motorcycle | f228 → f241 | 13f | 3.9 | 2.0x | (343, 155) | [png](032_id105_f228-241.png) |
| 33 | 311 | motorcycle | f541 → f552 | 11f | 3.9 | 2.9x | (328, 126) | [png](033_id311_f541-552.png) |
| 34 | 179 | motorcycle | f333 → f361 | 28f | 3.9 | 1.2x | (680, 432) | [png](034_id179_f333-361.png) |
| 35 | 198 | motorcycle | f390 → f411 | 21f | 3.9 | 2.0x | (643, 376) | [png](035_id198_f390-411.png) |
| 36 | 255 | motorcycle | f478 → f501 | 23f | 3.8 | 2.2x | (521, 392) | [png](036_id255_f478-501.png) |
| 37 | 132 | motorcycle | f343 → f357 | 14f | 2.4 | 3.8x | (631, 409) | [png](037_id132_f343-357.png) |
| 38 | 216 | motorcycle | f499 → f521 | 22f | 3.8 | 1.1x | (744, 432) | [png](038_id216_f499-521.png) |
| 39 | 109 | motorcycle | f254 → f281 | 27f | 3.8 | 1.3x | (775, 413) | [png](039_id109_f254-281.png) |
| 40 | 26 | motorcycle | f39 → f52 | 13f | 3.7 | 3.2x | (604, 278) | [png](040_id26_f39-52.png) |
| 41 | 40 | motorcycle | f61 → f83 | 22f | 3.7 | 1.5x | (684, 431) | [png](041_id40_f61-83.png) |
| 42 | 139 | motorcycle | f269 → f297 | 28f | 3.7 | 1.7x | (701, 360) | [png](042_id139_f269-297.png) |
| 43 | 183 | motorcycle | f387 → f416 | 29f | 3.7 | 1.2x | (446, 339) | [png](043_id183_f387-416.png) |
| 44 | 144 | motorcycle | f259 → f276 | 17f | 1.7 | 3.6x | (392, 282) | [png](044_id144_f259-276.png) |
| 45 | 229 | motorcycle | f474 → f496 | 22f | 3.6 | 2.5x | (666, 344) | [png](045_id229_f474-496.png) |
| 46 | 299 | motorcycle | f511 → f532 | 21f | 3.2 | 3.6x | (361, 304) | [png](046_id299_f511-532.png) |
| 47 | 25 | motorcycle | f73 → f81 | 8f | 3.6 | 2.6x | (370, 160) | [png](047_id25_f73-81.png) |
| 48 | 82 | motorcycle | f176 → f203 | 27f | 2.7 | 3.6x | (217, 325) | [png](048_id82_f176-203.png) |
| 49 | 283 | motorcycle | f477 → f502 | 25f | 3.6 | 2.4x | (579, 293) | [png](049_id283_f477-502.png) |
| 50 | 281 | motorcycle | f530 → f551 | 21f | 2.2 | 3.5x | (690, 410) | [png](050_id281_f530-551.png) |
| 51 | 318 | motorcycle | f555 → f563 | 8f | 3.4 | 2.6x | (322, 127) | [png](051_id318_f555-563.png) |
| 52 | 139 | motorcycle | f297 → f322 | 25f | 3.4 | 2.2x | (487, 252) | [png](052_id139_f297-322.png) |
| 53 | 96 | motorcycle | f247 → f276 | 29f | 3.4 | 3.3x | (705, 386) | [png](053_id96_f247-276.png) |
| 54 | 145 | motorcycle | f313 → f329 | 16f | 3.4 | 2.1x | (778, 412) | [png](054_id145_f313-329.png) |
| 55 | 189 | motorcycle | f351 → f374 | 23f | 3.3 | 1.3x | (719, 427) | [png](055_id189_f351-374.png) |
| 56 | 300 | motorcycle | f535 → f553 | 18f | 3.2 | 2.4x | (479, 251) | [png](056_id300_f535-553.png) |
| 57 | 103 | motorcycle | f220 → f242 | 22f | 3.2 | 1.3x | (741, 416) | [png](057_id103_f220-242.png) |
| 58 | 206 | motorcycle | f478 → f508 | 30f | 2.0 | 3.1x | (447, 421) | [png](058_id206_f478-508.png) |
| 59 | 85 | motorcycle | f332 → f351 | 19f | 3.0 | 1.1x | (759, 423) | [png](059_id85_f332-351.png) |
| 60 | 37 | motorcycle | f142 → f157 | 15f | 3.0 | 2.7x | (425, 222) | [png](060_id37_f142-157.png) |
| 61 | 127 | motorcycle | f205 → f222 | 17f | 0.7 | 2.9x | (266, 128) | [png](061_id127_f205-222.png) |
| 62 | 320 | motorcycle | f569 → f579 | 10f | 2.8 | 1.8x | (386, 117) | [png](062_id320_f569-579.png) |
| 63 | 96 | motorcycle | f445 → f467 | 22f | 2.8 | 2.6x | (525, 335) | [png](063_id96_f445-467.png) |
| 64 | 321 | motorcycle | f583 → f594 | 11f | 2.8 | 2.7x | (276, 131) | [png](064_id321_f583-594.png) |
| 65 | 245 | motorcycle | f462 → f476 | 14f | 2.8 | 1.4x | (673, 420) | [png](065_id245_f462-476.png) |
| 66 | 111 | motorcycle | f282 → f298 | 16f | 2.3 | 2.7x | (590, 303) | [png](066_id111_f282-298.png) |
| 67 | 250 | motorcycle | f515 → f540 | 25f | 2.7 | 2.2x | (754, 430) | [png](067_id250_f515-540.png) |
| 68 | 96 | motorcycle | f300 → f314 | 14f | 1.4 | 2.7x | (565, 371) | [png](068_id96_f300-314.png) |
| 69 | 203 | motorcycle | f396 → f416 | 20f | 2.7 | 1.4x | (697, 414) | [png](069_id203_f396-416.png) |
| 70 | 252 | motorcycle | f458 → f481 | 23f | 2.7 | 1.4x | (527, 425) | [png](070_id252_f458-481.png) |
| 71 | 194 | motorcycle | f365 → f388 | 23f | 2.7 | 1.9x | (654, 322) | [png](071_id194_f365-388.png) |
| 72 | 160 | motorcycle | f298 → f316 | 18f | 2.7 | 2.5x | (475, 358) | [png](072_id160_f298-316.png) |
| 73 | 299 | motorcycle | f533 → f549 | 16f | 1.2 | 2.7x | (388, 247) | [png](073_id299_f533-549.png) |
| 74 | 321 | motorcycle | f563 → f573 | 10f | 2.6 | 1.5x | (350, 116) | [png](074_id321_f563-573.png) |
| 75 | 25 | motorcycle | f24 → f31 | 7f | 2.6 | 1.6x | (421, 156) | [png](075_id25_f24-31.png) |
| 76 | 252 | motorcycle | f444 → f452 | 8f | 0.3 | 2.6x | (460, 283) | [png](076_id252_f444-452.png) |
| 77 | 137 | motorcycle | f294 → f316 | 22f | 2.6 | 2.2x | (597, 399) | [png](077_id137_f294-316.png) |
| 78 | 128 | motorcycle | f212 → f241 | 29f | 2.6 | 1.4x | (361, 438) | [png](078_id128_f212-241.png) |
| 79 | 267 | motorcycle | f528 → f536 | 8f | 1.2 | 2.5x | (372, 322) | [png](079_id267_f528-536.png) |
| 80 | 132 | motorcycle | f236 → f255 | 19f | 2.4 | 2.5x | (387, 209) | [png](080_id132_f236-255.png) |
| 81 | 96 | motorcycle | f419 → f445 | 26f | 1.6 | 2.5x | (627, 291) | [png](081_id96_f419-445.png) |
| 82 | 132 | motorcycle | f315 → f343 | 28f | 2.5 | 2.3x | (566, 306) | [png](082_id132_f315-343.png) |
| 83 | 126 | motorcycle | f208 → f224 | 16f | 0.6 | 2.5x | (402, 158) | [png](083_id126_f208-224.png) |
| 84 | 243 | motorcycle | f429 → f447 | 18f | 0.6 | 2.5x | (343, 161) | [png](084_id243_f429-447.png) |
| 85 | 270 | motorcycle | f482 → f510 | 28f | 2.5 | 1.5x | (309, 228) | [png](085_id270_f482-510.png) |
| 86 | 105 | motorcycle | f219 → f225 | 6f | 1.0 | 2.5x | (482, 266) | [png](086_id105_f219-225.png) |
| 87 | 317 | motorcycle | f560 → f574 | 14f | 1.4 | 2.5x | (325, 127) | [png](087_id317_f560-574.png) |
| 88 | 68 | motorcycle | f149 → f155 | 6f | 0.7 | 2.4x | (431, 159) | [png](088_id68_f149-155.png) |
| 89 | 260 | motorcycle | f431 → f446 | 15f | 2.1 | 2.4x | (444, 209) | [png](089_id260_f431-446.png) |
| 90 | 67 | motorcycle | f231 → f247 | 16f | 2.4 | 1.7x | (726, 424) | [png](090_id67_f231-247.png) |
| 91 | 255 | motorcycle | f456 → f478 | 22f | 2.4 | 1.2x | (696, 430) | [png](091_id255_f456-478.png) |
| 92 | 85 | motorcycle | f167 → f193 | 26f | 2.4 | 1.1x | (313, 153) | [png](092_id85_f167-193.png) |
| 93 | 96 | motorcycle | f365 → f385 | 20f | 1.1 | 2.3x | (625, 302) | [png](093_id96_f365-385.png) |
| 94 | 211 | motorcycle | f429 → f436 | 7f | 1.0 | 2.3x | (457, 303) | [png](094_id211_f429-436.png) |
| 95 | 255 | motorcycle | f437 → f456 | 19f | 2.3 | 1.6x | (639, 309) | [png](095_id255_f437-456.png) |
| 96 | 207 | motorcycle | f332 → f352 | 20f | 0.2 | 2.3x | (585, 409) | [png](096_id207_f332-352.png) |
| 97 | 158 | motorcycle | f280 → f296 | 16f | 2.1 | 2.3x | (370, 201) | [png](097_id158_f280-296.png) |
| 98 | 27 | motorcycle | f53 → f60 | 7f | 1.6 | 2.3x | (532, 245) | [png](098_id27_f53-60.png) |
| 99 | 173 | motorcycle | f375 → f391 | 16f | 2.3 | 1.8x | (586, 416) | [png](099_id173_f375-391.png) |
| 100 | 92 | motorcycle | f201 → f215 | 14f | 0.7 | 2.2x | (233, 336) | [png](100_id92_f201-215.png) |
| 101 | 259 | car | f452 → f472 | 20f | 0.2 | 2.2x | (786, 422) | [png](101_id259_f452-472.png) |
| 102 | 111 | motorcycle | f209 → f224 | 15f | 2.0 | 2.1x | (363, 159) | [png](102_id111_f209-224.png) |
| 103 | 227 | motorcycle | f457 → f465 | 8f | 0.5 | 2.1x | (436, 382) | [png](103_id227_f457-465.png) |
| 104 | 180 | motorcycle | f287 → f296 | 9f | 2.1 | 1.7x | (787, 382) | [png](104_id180_f287-296.png) |
| 105 | 193 | motorcycle | f339 → f354 | 15f | 0.7 | 2.0x | (546, 387) | [png](105_id193_f339-354.png) |

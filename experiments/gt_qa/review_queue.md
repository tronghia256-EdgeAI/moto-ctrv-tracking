# Review queue — segment `dense`

Sinh boi `python -m code.scripts.gt_review_queue` tu `data\annotations\NKKN_gt\dense\gt\gt.txt`.
27513 box · 335 track ID · 600 frame.

**190 cho can nhin** (thay vi tua het 600 frame) — uoc luong ~93 phut neu lam tuan tu.

> Danh sach nay **khong phai GT**. No chi noi *cho nao dang ngo*; dung hay sai van do ban quyet dinh khi nhin anh. Cho nao nhin thay pre-label da dung thi tick va di tiep — mot dong bi bo qua khong lam hong GT, mot dong sua bua thi co.


## Cach dung

1. Mo task `dense` trong CVAT.
2. Go so frame vao o frame (goc tren) de nhay thang toi cho can xem.
3. Cot **CVAT** la so go vao o do (CVAT dem tu 0, file MOT dem tu 1). Lan dau nen doi chieu 1 box de chac mapping dung.
4. Sua xong tick `[x]`. `Ctrl+S` moi 5-10 phut.


Quy doi frame: `CVAT = MOT - 1` · `frame tuyet doi trong video = MOT - 1 + 10700`


<sub>Nguong JUMP tu calibrate theo phan phoi cua chinh doan nay (median + 10xMAD): dich chuyen > 0.42 box/frame, dien tich doi > 2.3x. ORPHAN da bo qua 251 truong hop box < 40px — o kich thuoc do guide noi khong label.</sub>


## MERGE — track vo lam doi (viec chinh, lam truoc)

Moi dong = mot cap ID nghi la cung mot xe. Dung phim `M` hai lan. Day dung la thu C2 claim cai thien, nen sua het muc nay truoc khi lam gi khac.


- [ ] **frame 464** (MOT 465 · abs 11164) · motorcycle · conf 0.69  
      ID 232 chet o frame 464, ID 283 sinh o frame 466 (gap 2f, lech 0.2 box, cung class)  
      -> click box ID 232 -> M -> tua toi frame 466 -> click box ID 283 -> M
- [ ] **frame 393** (MOT 394 · abs 11093) · car · conf 0.52  
      ID 209 chet o frame 393, ID 241 sinh o frame 399 (gap 6f, lech 0.5 box, cung class)  
      -> click box ID 209 -> M -> tua toi frame 399 -> click box ID 241 -> M
- [ ] **frame 55** (MOT 56 · abs 10755) · car · conf 0.43  
      ID 24 chet o frame 55, ID 57 sinh o frame 62 (gap 7f, lech 0.4 box, cung class)  
      -> click box ID 24 -> M -> tua toi frame 62 -> click box ID 57 -> M
- [ ] **frame 594** (MOT 595 · abs 11294) · car · conf 0.36  
      ID 265 chet o frame 594, ID 335 sinh o frame 598 (gap 4f, lech 0.6 box, cung class)  
      -> click box ID 265 -> M -> tua toi frame 598 -> click box ID 335 -> M
- [ ] **frame 77** (MOT 78 · abs 10777) · motorcycle · conf 0.32  
      ID 35 chet o frame 77, ID 72 sinh o frame 82 (gap 5f, lech 0.9 box, cung class)  
      -> click box ID 35 -> M -> tua toi frame 82 -> click box ID 72 -> M
- [ ] **frame 57** (MOT 58 · abs 10757) · motorcycle · conf 0.32  
      ID 44 chet o frame 57, ID 56 sinh o frame 62 (gap 5f, lech 1.1 box, cung class)  
      -> click box ID 44 -> M -> tua toi frame 62 -> click box ID 56 -> M
- [ ] **frame 455** (MOT 456 · abs 11155) · motorcycle · conf 0.32  
      ID 183 chet o frame 455, ID 277 sinh o frame 461 (gap 6f, lech 1.2 box, cung class)  
      -> click box ID 183 -> M -> tua toi frame 461 -> click box ID 277 -> M
- [ ] **frame 296** (MOT 297 · abs 10996) · car · conf 0.31  
      ID 164 chet o frame 296, ID 200 sinh o frame 312 (gap 16f, lech 0.5 box, cung class)  
      -> click box ID 164 -> M -> tua toi frame 312 -> click box ID 200 -> M
- [ ] **frame 327** (MOT 328 · abs 11027) · motorcycle · conf 0.29  
      ID 176 chet o frame 327, ID 210 sinh o frame 332 (gap 5f, lech 1.3 box, cung class)  
      -> click box ID 176 -> M -> tua toi frame 332 -> click box ID 210 -> M
- [ ] **frame 585** (MOT 586 · abs 11285) · car · conf 0.28  
      ID 313 chet o frame 585, ID 331 sinh o frame 588 (gap 3f, lech 1.2 box, cung class)  
      -> click box ID 313 -> M -> tua toi frame 588 -> click box ID 331 -> M
- [ ] **frame 297** (MOT 298 · abs 10997) · motorcycle · conf 0.26  
      ID 158 chet o frame 297, ID 196 sinh o frame 306 (gap 9f, lech 1.1 box, cung class)  
      -> click box ID 158 -> M -> tua toi frame 306 -> click box ID 196 -> M
- [ ] **frame 262** (MOT 263 · abs 10962) · motorcycle · conf 0.26  
      ID 105 chet o frame 262, ID 171 sinh o frame 265 (gap 3f, lech 1.7 box, cung class)  
      -> click box ID 105 -> M -> tua toi frame 265 -> click box ID 171 -> M
- [ ] **frame 404** (MOT 405 · abs 11104) · motorcycle · conf 0.25  
      ID 225 chet o frame 404, ID 245 sinh o frame 406 (gap 2f, lech 1.3 box, cung class)  
      -> click box ID 225 -> M -> tua toi frame 406 -> click box ID 245 -> M
- [ ] **frame 505** (MOT 506 · abs 11205) · motorcycle · conf 0.25  
      ID 290 chet o frame 505, ID 300 sinh o frame 507 (gap 2f, lech 1.6 box, cung class)  
      -> click box ID 290 -> M -> tua toi frame 507 -> click box ID 300 -> M
- [ ] **frame 594** (MOT 595 · abs 11294) · motorcycle · conf 0.20  
      ID 321 chet o frame 594, ID 333 sinh o frame 598 (gap 4f, lech 0.6 box, KHAC class (motorcycle/car))  
      -> click box ID 321 -> M -> tua toi frame 598 -> click box ID 333 -> M
- [ ] **frame 567** (MOT 568 · abs 11267) · car · conf 0.20  
      ID 322 chet o frame 567, ID 327 sinh o frame 573 (gap 6f, lech 2.0 box, cung class)  
      -> click box ID 322 -> M -> tua toi frame 573 -> click box ID 327 -> M
- [ ] **frame 152** (MOT 153 · abs 10852) · motorcycle · conf 0.19  
      ID 58 chet o frame 152, ID 113 sinh o frame 162 (gap 10f, lech 1.6 box, cung class)  
      -> click box ID 58 -> M -> tua toi frame 162 -> click box ID 113 -> M
- [ ] **frame 462** (MOT 463 · abs 11162) · motorcycle · conf 0.18  
      ID 260 chet o frame 462, ID 281 sinh o frame 463 (gap 1f, lech 2.4 box, cung class)  
      -> click box ID 260 -> M -> tua toi frame 463 -> click box ID 281 -> M
- [ ] **frame 263** (MOT 264 · abs 10963) · car · conf 0.17  
      ID 152 chet o frame 263, ID 186 sinh o frame 280 (gap 17f, lech 1.2 box, cung class)  
      -> click box ID 152 -> M -> tua toi frame 280 -> click box ID 186 -> M
- [ ] **frame 294** (MOT 295 · abs 10994) · motorcycle · conf 0.17  
      ID 166 chet o frame 294, ID 201 sinh o frame 314 (gap 20f, lech 0.5 box, cung class)  
      -> click box ID 166 -> M -> tua toi frame 314 -> click box ID 201 -> M
- [ ] **frame 107** (MOT 108 · abs 10807) · motorcycle · conf 0.16  
      ID 66 chet o frame 107, ID 92 sinh o frame 128 (gap 21f, lech 1.1 box, cung class)  
      -> click box ID 66 -> M -> tua toi frame 128 -> click box ID 92 -> M
- [ ] **frame 140** (MOT 141 · abs 10840) · car · conf 0.16  
      ID 101 chet o frame 140, ID 106 sinh o frame 153 (gap 13f, lech 0.6 box, KHAC class (car/motorcycle))  
      -> click box ID 101 -> M -> tua toi frame 153 -> click box ID 106 -> M
- [ ] **frame 520** (MOT 521 · abs 11220) · motorcycle · conf 0.15  
      ID 253 chet o frame 520, ID 312 sinh o frame 534 (gap 14f, lech 1.3 box, cung class)  
      -> click box ID 253 -> M -> tua toi frame 534 -> click box ID 312 -> M
- [ ] **frame 496** (MOT 497 · abs 11196) · motorcycle · conf 0.15  
      ID 171 chet o frame 496, ID 295 sinh o frame 499 (gap 3f, lech 2.6 box, cung class)  
      -> click box ID 171 -> M -> tua toi frame 499 -> click box ID 295 -> M
- [ ] **frame 236** (MOT 237 · abs 10936) · car · conf 0.14  
      ID 150 chet o frame 236, ID 152 sinh o frame 242 (gap 6f, lech 1.5 box, cung class)  
      -> click box ID 150 -> M -> tua toi frame 242 -> click box ID 152 -> M
- [ ] **frame 494** (MOT 495 · abs 11194) · car · conf 0.14  
      ID 286 chet o frame 494, ID 297 sinh o frame 502 (gap 8f, lech 0.7 box, KHAC class (car/motorcycle))  
      -> click box ID 286 -> M -> tua toi frame 502 -> click box ID 297 -> M
- [ ] **frame 312** (MOT 313 · abs 11012) · car · conf 0.13  
      ID 186 chet o frame 312, ID 206 sinh o frame 328 (gap 16f, lech 0.8 box, KHAC class (car/motorcycle))  
      -> click box ID 186 -> M -> tua toi frame 328 -> click box ID 206 -> M
- [ ] **frame 355** (MOT 356 · abs 11055) · motorcycle · conf 0.12  
      ID 218 chet o frame 355, ID 225 sinh o frame 375 (gap 20f, lech 1.3 box, cung class)  
      -> click box ID 218 -> M -> tua toi frame 375 -> click box ID 225 -> M
- [ ] **frame 552** (MOT 553 · abs 11252) · motorcycle · conf 0.11  
      ID 272 chet o frame 552, ID 323 sinh o frame 560 (gap 8f, lech 1.9 box, cung class)  
      -> click box ID 272 -> M -> tua toi frame 560 -> click box ID 323 -> M

## SPLIT — mot ID gom nhieu xe khac nhau (nguy hiem nhat)

Nguoc voi MERGE: mot ID nhay sang xe khac sau khi bi che. Loai nay nguy hiem nhat cho ablation — GT gop nham hai xe se **thuong** cho tracker nao dam noi qua cho che khuat, tuc la thoi phong dung cai ma C2 claim. Nhin ky hai frame roi Split neu la hai xe.


*74 muc, hien 60 muc dang ngo nhat (`--per-type` de xem them).*


- [ ] **frame 48** (MOT 49 · abs 10748) · motorcycle · conf 1.00  
      ID 35 bien mat o frame 18 roi hien lai o 48 (cach 30f): nhay 7.6 box qua khoang trong, kich thuoc doi 7.2x  
      -> so sanh xe o 18 va 48: neu la HAI xe khac nhau -> Split track tai 48 thanh 2 ID
- [ ] **frame 135** (MOT 136 · abs 10835) · motorcycle · conf 1.00  
      ID 37 bien mat o frame 111 roi hien lai o 135 (cach 24f): nhay 4.7 box qua khoang trong, kich thuoc doi 9.7x  
      -> so sanh xe o 111 va 135: neu la HAI xe khac nhau -> Split track tai 135 thanh 2 ID
- [ ] **frame 282** (MOT 283 · abs 10982) · motorcycle · conf 1.00  
      ID 82 bien mat o frame 271 roi hien lai o 282 (cach 11f): nhay 3.3 box qua khoang trong, kich thuoc doi 8.5x  
      -> so sanh xe o 271 va 282: neu la HAI xe khac nhau -> Split track tai 282 thanh 2 ID
- [ ] **frame 159** (MOT 160 · abs 10859) · motorcycle · conf 1.00  
      ID 85 bien mat o frame 134 roi hien lai o 159 (cach 25f): nhay 6.9 box qua khoang trong, kich thuoc doi 2.2x  
      -> so sanh xe o 134 va 159: neu la HAI xe khac nhau -> Split track tai 159 thanh 2 ID
- [ ] **frame 300** (MOT 301 · abs 11000) · motorcycle · conf 1.00  
      ID 96 bien mat o frame 276 roi hien lai o 300 (cach 24f): kich thuoc doi 6.1x  
      -> so sanh xe o 276 va 300: neu la HAI xe khac nhau -> Split track tai 300 thanh 2 ID
- [ ] **frame 259** (MOT 260 · abs 10959) · motorcycle · conf 1.00  
      ID 105 bien mat o frame 241 roi hien lai o 259 (cach 18f): nhay 9.1 box qua khoang trong  
      -> so sanh xe o 241 va 259: neu la HAI xe khac nhau -> Split track tai 259 thanh 2 ID
- [ ] **frame 272** (MOT 273 · abs 10972) · motorcycle · conf 1.00  
      ID 107 bien mat o frame 247 roi hien lai o 272 (cach 25f): nhay 11.1 box qua khoang trong, kich thuoc doi 13.7x  
      -> so sanh xe o 247 va 272: neu la HAI xe khac nhau -> Split track tai 272 thanh 2 ID
- [ ] **frame 324** (MOT 325 · abs 11024) · motorcycle · conf 1.00  
      ID 180 bien mat o frame 296 roi hien lai o 324 (cach 28f): nhay 6.0 box qua khoang trong  
      -> so sanh xe o 296 va 324: neu la HAI xe khac nhau -> Split track tai 324 thanh 2 ID
- [ ] **frame 335** (MOT 336 · abs 11035) · motorcycle · conf 1.00  
      ID 208 bien mat o frame 329 roi hien lai o 335 (cach 6f): kich thuoc doi 7.0x  
      -> so sanh xe o 329 va 335: neu la HAI xe khac nhau -> Split track tai 335 thanh 2 ID
- [ ] **frame 392** (MOT 393 · abs 11092) · motorcycle · conf 1.00  
      ID 225 bien mat o frame 378 roi hien lai o 392 (cach 14f): nhay 9.3 box qua khoang trong, kich thuoc doi 2.7x  
      -> so sanh xe o 378 va 392: neu la HAI xe khac nhau -> Split track tai 392 thanh 2 ID
- [ ] **frame 556** (MOT 557 · abs 11256) · motorcycle · conf 1.00  
      ID 276 bien mat o frame 537 roi hien lai o 556 (cach 19f): kich thuoc doi 7.1x  
      -> so sanh xe o 537 va 556: neu la HAI xe khac nhau -> Split track tai 556 thanh 2 ID
- [ ] **frame 151** (MOT 152 · abs 10851) · motorcycle · conf 0.96  
      ID 22 bien mat o frame 127 roi hien lai o 151 (cach 24f): nhay 5.8 box qua khoang trong  
      -> so sanh xe o 127 va 151: neu la HAI xe khac nhau -> Split track tai 151 thanh 2 ID
- [ ] **frame 122** (MOT 123 · abs 10822) · motorcycle · conf 0.93  
      ID 23 bien mat o frame 99 roi hien lai o 122 (cach 23f): nhay 5.7 box qua khoang trong  
      -> so sanh xe o 99 va 122: neu la HAI xe khac nhau -> Split track tai 122 thanh 2 ID
- [ ] **frame 394** (MOT 395 · abs 11094) · motorcycle · conf 0.84  
      ID 174 bien mat o frame 375 roi hien lai o 394 (cach 19f): nhay 5.3 box qua khoang trong, kich thuoc doi 2.6x  
      -> so sanh xe o 375 va 394: neu la HAI xe khac nhau -> Split track tai 394 thanh 2 ID
- [ ] **frame 73** (MOT 74 · abs 10773) · motorcycle · conf 0.78  
      ID 25 bien mat o frame 47 roi hien lai o 73 (cach 26f): nhay 5.1 box qua khoang trong, kich thuoc doi 4.6x  
      -> so sanh xe o 47 va 73: neu la HAI xe khac nhau -> Split track tai 73 thanh 2 ID
- [ ] **frame 108** (MOT 109 · abs 10808) · motorcycle · conf 0.76  
      ID 49 bien mat o frame 91 roi hien lai o 108 (cach 17f): nhay 5.0 box qua khoang trong  
      -> so sanh xe o 91 va 108: neu la HAI xe khac nhau -> Split track tai 108 thanh 2 ID
- [ ] **frame 231** (MOT 232 · abs 10931) · motorcycle · conf 0.73  
      ID 102 bien mat o frame 215 roi hien lai o 231 (cach 16f): nhay 2.7 box qua khoang trong, kich thuoc doi 4.9x  
      -> so sanh xe o 215 va 231: neu la HAI xe khac nhau -> Split track tai 231 thanh 2 ID
- [ ] **frame 286** (MOT 287 · abs 10986) · motorcycle · conf 0.67  
      ID 136 bien mat o frame 258 roi hien lai o 286 (cach 28f): nhay 4.7 box qua khoang trong  
      -> so sanh xe o 258 va 286: neu la HAI xe khac nhau -> Split track tai 286 thanh 2 ID
- [ ] **frame 423** (MOT 424 · abs 11123) · motorcycle · conf 0.57  
      ID 215 bien mat o frame 407 roi hien lai o 423 (cach 16f): nhay 4.3 box qua khoang trong  
      -> so sanh xe o 407 va 423: neu la HAI xe khac nhau -> Split track tai 423 thanh 2 ID
- [ ] **frame 498** (MOT 499 · abs 11198) · motorcycle · conf 0.56  
      ID 247 bien mat o frame 481 roi hien lai o 498 (cach 17f): nhay 2.5 box qua khoang trong, kich thuoc doi 4.3x  
      -> so sanh xe o 481 va 498: neu la HAI xe khac nhau -> Split track tai 498 thanh 2 ID
- [ ] **frame 387** (MOT 388 · abs 11087) · motorcycle · conf 0.55  
      ID 183 bien mat o frame 360 roi hien lai o 387 (cach 27f): nhay 4.2 box qua khoang trong  
      -> so sanh xe o 360 va 387: neu la HAI xe khac nhau -> Split track tai 387 thanh 2 ID
- [ ] **frame 89** (MOT 90 · abs 10789) · motorcycle · conf 0.54  
      ID 55 bien mat o frame 70 roi hien lai o 89 (cach 19f): nhay 4.2 box qua khoang trong, kich thuoc doi 2.3x  
      -> so sanh xe o 70 va 89: neu la HAI xe khac nhau -> Split track tai 89 thanh 2 ID
- [ ] **frame 246** (MOT 247 · abs 10946) · motorcycle · conf 0.53  
      ID 116 bien mat o frame 232 roi hien lai o 246 (cach 14f): nhay 2.3 box qua khoang trong, kich thuoc doi 4.1x  
      -> so sanh xe o 232 va 246: neu la HAI xe khac nhau -> Split track tai 246 thanh 2 ID
- [ ] **frame 507** (MOT 508 · abs 11207) · motorcycle · conf 0.50  
      ID 248 bien mat o frame 483 roi hien lai o 507 (cach 24f): nhay 4.0 box qua khoang trong  
      -> so sanh xe o 483 va 507: neu la HAI xe khac nhau -> Split track tai 507 thanh 2 ID
- [ ] **frame 552** (MOT 553 · abs 11252) · motorcycle · conf 0.47  
      ID 311 bien mat o frame 541 roi hien lai o 552 (cach 11f): nhay 3.9 box qua khoang trong, kich thuoc doi 2.9x  
      -> so sanh xe o 541 va 552: neu la HAI xe khac nhau -> Split track tai 552 thanh 2 ID
- [ ] **frame 361** (MOT 362 · abs 11061) · motorcycle · conf 0.46  
      ID 179 bien mat o frame 333 roi hien lai o 361 (cach 28f): nhay 3.9 box qua khoang trong  
      -> so sanh xe o 333 va 361: neu la HAI xe khac nhau -> Split track tai 361 thanh 2 ID
- [ ] **frame 411** (MOT 412 · abs 11111) · motorcycle · conf 0.46  
      ID 198 bien mat o frame 390 roi hien lai o 411 (cach 21f): nhay 3.9 box qua khoang trong, kich thuoc doi 2.0x  
      -> so sanh xe o 390 va 411: neu la HAI xe khac nhau -> Split track tai 411 thanh 2 ID
- [ ] **frame 501** (MOT 502 · abs 11201) · motorcycle · conf 0.46  
      ID 255 bien mat o frame 478 roi hien lai o 501 (cach 23f): nhay 3.8 box qua khoang trong, kich thuoc doi 2.2x  
      -> so sanh xe o 478 va 501: neu la HAI xe khac nhau -> Split track tai 501 thanh 2 ID
- [ ] **frame 357** (MOT 358 · abs 11057) · motorcycle · conf 0.45  
      ID 132 bien mat o frame 343 roi hien lai o 357 (cach 14f): nhay 2.4 box qua khoang trong, kich thuoc doi 3.8x  
      -> so sanh xe o 343 va 357: neu la HAI xe khac nhau -> Split track tai 357 thanh 2 ID
- [ ] **frame 521** (MOT 522 · abs 11221) · motorcycle · conf 0.45  
      ID 216 bien mat o frame 499 roi hien lai o 521 (cach 22f): nhay 3.8 box qua khoang trong  
      -> so sanh xe o 499 va 521: neu la HAI xe khac nhau -> Split track tai 521 thanh 2 ID
- [ ] **frame 281** (MOT 282 · abs 10981) · motorcycle · conf 0.44  
      ID 109 bien mat o frame 254 roi hien lai o 281 (cach 27f): nhay 3.8 box qua khoang trong  
      -> so sanh xe o 254 va 281: neu la HAI xe khac nhau -> Split track tai 281 thanh 2 ID
- [ ] **frame 52** (MOT 53 · abs 10752) · motorcycle · conf 0.43  
      ID 26 bien mat o frame 39 roi hien lai o 52 (cach 13f): nhay 3.7 box qua khoang trong, kich thuoc doi 3.2x  
      -> so sanh xe o 39 va 52: neu la HAI xe khac nhau -> Split track tai 52 thanh 2 ID
- [ ] **frame 83** (MOT 84 · abs 10783) · motorcycle · conf 0.43  
      ID 40 bien mat o frame 61 roi hien lai o 83 (cach 22f): nhay 3.7 box qua khoang trong  
      -> so sanh xe o 61 va 83: neu la HAI xe khac nhau -> Split track tai 83 thanh 2 ID
- [ ] **frame 297** (MOT 298 · abs 10997) · motorcycle · conf 0.43  
      ID 139 bien mat o frame 269 roi hien lai o 297 (cach 28f): nhay 3.7 box qua khoang trong  
      -> so sanh xe o 269 va 297: neu la HAI xe khac nhau -> Split track tai 297 thanh 2 ID
- [ ] **frame 276** (MOT 277 · abs 10976) · motorcycle · conf 0.41  
      ID 144 bien mat o frame 259 roi hien lai o 276 (cach 17f): kich thuoc doi 3.6x  
      -> so sanh xe o 259 va 276: neu la HAI xe khac nhau -> Split track tai 276 thanh 2 ID
- [ ] **frame 496** (MOT 497 · abs 11196) · motorcycle · conf 0.41  
      ID 229 bien mat o frame 474 roi hien lai o 496 (cach 22f): nhay 3.6 box qua khoang trong, kich thuoc doi 2.5x  
      -> so sanh xe o 474 va 496: neu la HAI xe khac nhau -> Split track tai 496 thanh 2 ID
- [ ] **frame 532** (MOT 533 · abs 11232) · motorcycle · conf 0.40  
      ID 299 bien mat o frame 511 roi hien lai o 532 (cach 21f): nhay 3.2 box qua khoang trong, kich thuoc doi 3.6x  
      -> so sanh xe o 511 va 532: neu la HAI xe khac nhau -> Split track tai 532 thanh 2 ID
- [ ] **frame 502** (MOT 503 · abs 11202) · motorcycle · conf 0.39  
      ID 283 bien mat o frame 477 roi hien lai o 502 (cach 25f): nhay 3.6 box qua khoang trong, kich thuoc doi 2.4x  
      -> so sanh xe o 477 va 502: neu la HAI xe khac nhau -> Split track tai 502 thanh 2 ID
- [ ] **frame 551** (MOT 552 · abs 11251) · motorcycle · conf 0.39  
      ID 281 bien mat o frame 530 roi hien lai o 551 (cach 21f): nhay 2.2 box qua khoang trong, kich thuoc doi 3.5x  
      -> so sanh xe o 530 va 551: neu la HAI xe khac nhau -> Split track tai 551 thanh 2 ID
- [ ] **frame 563** (MOT 564 · abs 11263) · motorcycle · conf 0.36  
      ID 318 bien mat o frame 555 roi hien lai o 563 (cach 8f): nhay 3.4 box qua khoang trong, kich thuoc doi 2.6x  
      -> so sanh xe o 555 va 563: neu la HAI xe khac nhau -> Split track tai 563 thanh 2 ID
- [ ] **frame 329** (MOT 330 · abs 11029) · motorcycle · conf 0.36  
      ID 145 bien mat o frame 313 roi hien lai o 329 (cach 16f): nhay 3.4 box qua khoang trong, kich thuoc doi 2.1x  
      -> so sanh xe o 313 va 329: neu la HAI xe khac nhau -> Split track tai 329 thanh 2 ID
- [ ] **frame 374** (MOT 375 · abs 11074) · motorcycle · conf 0.33  
      ID 189 bien mat o frame 351 roi hien lai o 374 (cach 23f): nhay 3.3 box qua khoang trong  
      -> so sanh xe o 351 va 374: neu la HAI xe khac nhau -> Split track tai 374 thanh 2 ID
- [ ] **frame 553** (MOT 554 · abs 11253) · motorcycle · conf 0.31  
      ID 300 bien mat o frame 535 roi hien lai o 553 (cach 18f): nhay 3.2 box qua khoang trong, kich thuoc doi 2.4x  
      -> so sanh xe o 535 va 553: neu la HAI xe khac nhau -> Split track tai 553 thanh 2 ID
- [ ] **frame 242** (MOT 243 · abs 10942) · motorcycle · conf 0.29  
      ID 103 bien mat o frame 220 roi hien lai o 242 (cach 22f): nhay 3.2 box qua khoang trong  
      -> so sanh xe o 220 va 242: neu la HAI xe khac nhau -> Split track tai 242 thanh 2 ID
- [ ] **frame 508** (MOT 509 · abs 11208) · motorcycle · conf 0.27  
      ID 206 bien mat o frame 478 roi hien lai o 508 (cach 30f): kich thuoc doi 3.1x  
      -> so sanh xe o 478 va 508: neu la HAI xe khac nhau -> Split track tai 508 thanh 2 ID
- [ ] **frame 222** (MOT 223 · abs 10922) · motorcycle · conf 0.22  
      ID 127 bien mat o frame 205 roi hien lai o 222 (cach 17f): kich thuoc doi 2.9x  
      -> so sanh xe o 205 va 222: neu la HAI xe khac nhau -> Split track tai 222 thanh 2 ID
- [ ] **frame 579** (MOT 580 · abs 11279) · motorcycle · conf 0.21  
      ID 320 bien mat o frame 569 roi hien lai o 579 (cach 10f): nhay 2.8 box qua khoang trong  
      -> so sanh xe o 569 va 579: neu la HAI xe khac nhau -> Split track tai 579 thanh 2 ID
- [ ] **frame 594** (MOT 595 · abs 11294) · motorcycle · conf 0.20  
      ID 321 bien mat o frame 583 roi hien lai o 594 (cach 11f): nhay 2.8 box qua khoang trong, kich thuoc doi 2.7x  
      -> so sanh xe o 583 va 594: neu la HAI xe khac nhau -> Split track tai 594 thanh 2 ID
- [ ] **frame 476** (MOT 477 · abs 11176) · motorcycle · conf 0.20  
      ID 245 bien mat o frame 462 roi hien lai o 476 (cach 14f): nhay 2.8 box qua khoang trong  
      -> so sanh xe o 462 va 476: neu la HAI xe khac nhau -> Split track tai 476 thanh 2 ID
- [ ] **frame 298** (MOT 299 · abs 10998) · motorcycle · conf 0.18  
      ID 111 bien mat o frame 282 roi hien lai o 298 (cach 16f): nhay 2.3 box qua khoang trong, kich thuoc doi 2.7x  
      -> so sanh xe o 282 va 298: neu la HAI xe khac nhau -> Split track tai 298 thanh 2 ID
- [ ] **frame 540** (MOT 541 · abs 11240) · motorcycle · conf 0.18  
      ID 250 bien mat o frame 515 roi hien lai o 540 (cach 25f): nhay 2.7 box qua khoang trong, kich thuoc doi 2.2x  
      -> so sanh xe o 515 va 540: neu la HAI xe khac nhau -> Split track tai 540 thanh 2 ID
- [ ] **frame 416** (MOT 417 · abs 11116) · motorcycle · conf 0.17  
      ID 203 bien mat o frame 396 roi hien lai o 416 (cach 20f): nhay 2.7 box qua khoang trong  
      -> so sanh xe o 396 va 416: neu la HAI xe khac nhau -> Split track tai 416 thanh 2 ID
- [ ] **frame 481** (MOT 482 · abs 11181) · motorcycle · conf 0.17  
      ID 252 bien mat o frame 458 roi hien lai o 481 (cach 23f): nhay 2.7 box qua khoang trong  
      -> so sanh xe o 458 va 481: neu la HAI xe khac nhau -> Split track tai 481 thanh 2 ID
- [ ] **frame 388** (MOT 389 · abs 11088) · motorcycle · conf 0.17  
      ID 194 bien mat o frame 365 roi hien lai o 388 (cach 23f): nhay 2.7 box qua khoang trong  
      -> so sanh xe o 365 va 388: neu la HAI xe khac nhau -> Split track tai 388 thanh 2 ID
- [ ] **frame 316** (MOT 317 · abs 11016) · motorcycle · conf 0.17  
      ID 160 bien mat o frame 298 roi hien lai o 316 (cach 18f): nhay 2.7 box qua khoang trong, kich thuoc doi 2.5x  
      -> so sanh xe o 298 va 316: neu la HAI xe khac nhau -> Split track tai 316 thanh 2 ID
- [ ] **frame 316** (MOT 317 · abs 11016) · motorcycle · conf 0.15  
      ID 137 bien mat o frame 294 roi hien lai o 316 (cach 22f): nhay 2.6 box qua khoang trong, kich thuoc doi 2.2x  
      -> so sanh xe o 294 va 316: neu la HAI xe khac nhau -> Split track tai 316 thanh 2 ID
- [ ] **frame 241** (MOT 242 · abs 10941) · motorcycle · conf 0.15  
      ID 128 bien mat o frame 212 roi hien lai o 241 (cach 29f): nhay 2.6 box qua khoang trong  
      -> so sanh xe o 212 va 241: neu la HAI xe khac nhau -> Split track tai 241 thanh 2 ID
- [ ] **frame 536** (MOT 537 · abs 11236) · motorcycle · conf 0.13  
      ID 267 bien mat o frame 528 roi hien lai o 536 (cach 8f): kich thuoc doi 2.5x  
      -> so sanh xe o 528 va 536: neu la HAI xe khac nhau -> Split track tai 536 thanh 2 ID
- [ ] **frame 224** (MOT 225 · abs 10924) · motorcycle · conf 0.13  
      ID 126 bien mat o frame 208 roi hien lai o 224 (cach 16f): kich thuoc doi 2.5x  
      -> so sanh xe o 208 va 224: neu la HAI xe khac nhau -> Split track tai 224 thanh 2 ID
- [ ] **frame 447** (MOT 448 · abs 11147) · motorcycle · conf 0.12  
      ID 243 bien mat o frame 429 roi hien lai o 447 (cach 18f): kich thuoc doi 2.5x  
      -> so sanh xe o 429 va 447: neu la HAI xe khac nhau -> Split track tai 447 thanh 2 ID

## JUMP — box cua mot ID doi dot ngot (playback kho thay)

Track khong dut, nhung box di chuyen/doi kich thuoc bat thuong so voi chinh doan nay. Hai kha nang, phai nhin moi biet: (a) ID that su nhay sang xe khac -> Split roi merge lai; (b) box chi bi che nen thu lai -> keo box cho khit. Vi khong co birth/death nen tua playback gan nhu khong bao gio thay loai nay.


- [ ] **frame 86** (MOT 87 · abs 10786) · motorcycle · conf 1.00  
      ID 9 tai frame 86: nhay 2.98 box/frame (nguong 0.42), dien tich doi 2.9x (nguong 2.3)  
      -> xem ID 9 truoc/sau frame 86: box nhay sang xe khac -> Split roi merge lai; box chi bi che/lech -> keo lai cho khit
- [ ] **frame 136** (MOT 137 · abs 10836) · motorcycle · conf 1.00  
      ID 37 tai frame 136: nhay 7.25 box/frame (nguong 0.42)  
      -> xem ID 37 truoc/sau frame 136: box nhay sang xe khac -> Split roi merge lai; box chi bi che/lech -> keo lai cho khit
- [ ] **frame 123** (MOT 124 · abs 10823) · motorcycle · conf 0.37  
      ID 63 tai frame 123: nhay 0.73 box/frame (nguong 0.42)  
      -> xem ID 63 truoc/sau frame 123: box nhay sang xe khac -> Split roi merge lai; box chi bi che/lech -> keo lai cho khit
- [ ] **frame 392** (MOT 393 · abs 11092) · motorcycle · conf 0.29  
      ID 225 tai frame 392: nhay 0.67 box/frame (nguong 0.42) (cach frame truoc 14f)  
      -> xem ID 225 truoc/sau frame 392: box nhay sang xe khac -> Split roi merge lai; box chi bi che/lech -> keo lai cho khit
- [ ] **frame 46** (MOT 47 · abs 10746) · motorcycle · conf 0.18  
      ID 39 tai frame 46: nhay 0.57 box/frame (nguong 0.42)  
      -> xem ID 39 truoc/sau frame 46: box nhay sang xe khac -> Split roi merge lai; box chi bi che/lech -> keo lai cho khit
- [ ] **frame 109** (MOT 110 · abs 10809) · motorcycle · conf 0.17  
      ID 49 tai frame 109: nhay 0.56 box/frame (nguong 0.42)  
      -> xem ID 49 truoc/sau frame 109: box nhay sang xe khac -> Split roi merge lai; box chi bi che/lech -> keo lai cho khit
- [ ] **frame 262** (MOT 263 · abs 10962) · motorcycle · conf 0.13  
      ID 105 tai frame 262: nhay 0.53 box/frame (nguong 0.42) (cach frame truoc 3f)  
      -> xem ID 105 truoc/sau frame 262: box nhay sang xe khac -> Split roi merge lai; box chi bi che/lech -> keo lai cho khit
- [ ] **frame 513** (MOT 514 · abs 11213) · motorcycle · conf 0.12  
      ID 268 tai frame 513: dien tich doi 2.9x (nguong 2.3)  
      -> xem ID 268 truoc/sau frame 513: box nhay sang xe khac -> Split roi merge lai; box chi bi che/lech -> keo lai cho khit
- [ ] **frame 582** (MOT 583 · abs 11282) · motorcycle · conf 0.11  
      ID 318 tai frame 582: nhay 0.51 box/frame (nguong 0.42)  
      -> xem ID 318 truoc/sau frame 582: box nhay sang xe khac -> Split roi merge lai; box chi bi che/lech -> keo lai cho khit
- [ ] **frame 10** (MOT 11 · abs 10710) · motorcycle · conf 0.11  
      ID 17 tai frame 10: nhay 0.51 box/frame (nguong 0.42)  
      -> xem ID 17 truoc/sau frame 10: box nhay sang xe khac -> Split roi merge lai; box chi bi che/lech -> keo lai cho khit
- [ ] **frame 290** (MOT 291 · abs 10990) · motorcycle · conf 0.10  
      ID 82 tai frame 290: nhay 0.51 box/frame (nguong 0.42), dien tich doi 2.3x (nguong 2.3) (cach frame truoc 2f)  
      -> xem ID 82 truoc/sau frame 290: box nhay sang xe khac -> Split roi merge lai; box chi bi che/lech -> keo lai cho khit
- [ ] **frame 16** (MOT 17 · abs 10716) · car · conf 0.10  
      ID 24 tai frame 16: nhay 0.50 box/frame (nguong 0.42)  
      -> xem ID 24 truoc/sau frame 16: box nhay sang xe khac -> Split roi merge lai; box chi bi che/lech -> keo lai cho khit
- [ ] **frame 286** (MOT 287 · abs 10986) · motorcycle · conf 0.08  
      ID 126 tai frame 286: nhay 0.49 box/frame (nguong 0.42)  
      -> xem ID 126 truoc/sau frame 286: box nhay sang xe khac -> Split roi merge lai; box chi bi che/lech -> keo lai cho khit
- [ ] **frame 281** (MOT 282 · abs 10981) · motorcycle · conf 0.06  
      ID 121 tai frame 281: dien tich doi 2.6x (nguong 2.3)  
      -> xem ID 121 truoc/sau frame 281: box nhay sang xe khac -> Split roi merge lai; box chi bi che/lech -> keo lai cho khit
- [ ] **frame 73** (MOT 74 · abs 10773) · motorcycle · conf 0.05  
      ID 26 tai frame 73: dien tich doi 2.5x (nguong 2.3) (cach frame truoc 2f)  
      -> xem ID 26 truoc/sau frame 73: box nhay sang xe khac -> Split roi merge lai; box chi bi che/lech -> keo lai cho khit
- [ ] **frame 51** (MOT 52 · abs 10751) · motorcycle · conf 0.03  
      ID 5 tai frame 51: dien tich doi 2.5x (nguong 2.3)  
      -> xem ID 5 truoc/sau frame 51: box nhay sang xe khac -> Split roi merge lai; box chi bi che/lech -> keo lai cho khit
- [ ] **frame 81** (MOT 82 · abs 10781) · motorcycle · conf 0.03  
      ID 25 tai frame 81: nhay 0.45 box/frame (nguong 0.42) (cach frame truoc 8f)  
      -> xem ID 25 truoc/sau frame 81: box nhay sang xe khac -> Split roi merge lai; box chi bi che/lech -> keo lai cho khit
- [ ] **frame 272** (MOT 273 · abs 10972) · motorcycle · conf 0.03  
      ID 107 tai frame 272: nhay 0.44 box/frame (nguong 0.42) (cach frame truoc 25f)  
      -> xem ID 107 truoc/sau frame 272: box nhay sang xe khac -> Split roi merge lai; box chi bi che/lech -> keo lai cho khit
- [ ] **frame 163** (MOT 164 · abs 10863) · motorcycle · conf 0.02  
      ID 78 tai frame 163: nhay 0.44 box/frame (nguong 0.42)  
      -> xem ID 78 truoc/sau frame 163: box nhay sang xe khac -> Split roi merge lai; box chi bi che/lech -> keo lai cho khit
- [ ] **frame 119** (MOT 120 · abs 10819) · motorcycle · conf 0.02  
      ID 60 tai frame 119: dien tich doi 2.4x (nguong 2.3)  
      -> xem ID 60 truoc/sau frame 119: box nhay sang xe khac -> Split roi merge lai; box chi bi che/lech -> keo lai cho khit
- [ ] **frame 142** (MOT 143 · abs 10842) · motorcycle · conf 0.01  
      ID 77 tai frame 142: nhay 0.43 box/frame (nguong 0.42)  
      -> xem ID 77 truoc/sau frame 142: box nhay sang xe khac -> Split roi merge lai; box chi bi che/lech -> keo lai cho khit
- [ ] **frame 180** (MOT 181 · abs 10880) · motorcycle · conf 0.01  
      ID 110 tai frame 180: nhay 0.43 box/frame (nguong 0.42)  
      -> xem ID 110 truoc/sau frame 180: box nhay sang xe khac -> Split roi merge lai; box chi bi che/lech -> keo lai cho khit
- [ ] **frame 295** (MOT 296 · abs 10995) · motorcycle · conf 0.00  
      ID 182 tai frame 295: dien tich doi 2.3x (nguong 2.3)  
      -> xem ID 182 truoc/sau frame 295: box nhay sang xe khac -> Split roi merge lai; box chi bi che/lech -> keo lai cho khit
- [ ] **frame 148** (MOT 149 · abs 10848) · motorcycle · conf 0.00  
      ID 100 tai frame 148: nhay 0.42 box/frame (nguong 0.42)  
      -> xem ID 100 truoc/sau frame 148: box nhay sang xe khac -> Split roi merge lai; box chi bi che/lech -> keo lai cho khit
- [ ] **frame 140** (MOT 141 · abs 10840) · motorcycle · conf 0.00  
      ID 99 tai frame 140: dien tich doi 2.3x (nguong 2.3) (cach frame truoc 2f)  
      -> xem ID 99 truoc/sau frame 140: box nhay sang xe khac -> Split roi merge lai; box chi bi che/lech -> keo lai cho khit

## DUP — hai ID dinh cung mot xe

Hai ID cung trum mot xe. Neu dung la mot xe, xoa bot mot track — de nguyen se thanh FP trong moi frame.


- [ ] **frame 150** (MOT 151 · abs 10850) · motorcycle · conf 0.84  
      ID 58 va ID 82 chong nhau IoU 0.84 trong 3 frame, tu frame 150  
      -> neu la 1 xe: xoa track thua (hoac merge); neu that su 2 xe che nhau: giu nguyen
- [ ] **frame 84** (MOT 85 · abs 10784) · motorcycle · conf 0.78  
      ID 9 va ID 59 chong nhau IoU 0.78 trong 2 frame, tu frame 84  
      -> neu la 1 xe: xoa track thua (hoac merge); neu that su 2 xe che nhau: giu nguyen
- [ ] **frame 550** (MOT 551 · abs 11250) · motorcycle · conf 0.75  
      ID 291 va ID 299 chong nhau IoU 0.75 trong 3 frame, tu frame 550  
      -> neu la 1 xe: xoa track thua (hoac merge); neu that su 2 xe che nhau: giu nguyen
- [ ] **frame 395** (MOT 396 · abs 11095) · car · conf 0.73  
      ID 131 va ID 226 chong nhau IoU 0.73 trong 59 frame, tu frame 395  
      -> neu la 1 xe: xoa track thua (hoac merge); neu that su 2 xe che nhau: giu nguyen
- [ ] **frame 219** (MOT 220 · abs 10919) · motorcycle · conf 0.72  
      ID 105 va ID 116 chong nhau IoU 0.72 trong 1 frame, tu frame 219  
      -> neu la 1 xe: xoa track thua (hoac merge); neu that su 2 xe che nhau: giu nguyen
- [ ] **frame 416** (MOT 417 · abs 11116) · motorcycle · conf 0.69  
      ID 183 va ID 219 chong nhau IoU 0.69 trong 2 frame, tu frame 416  
      -> neu la 1 xe: xoa track thua (hoac merge); neu that su 2 xe che nhau: giu nguyen
- [ ] **frame 401** (MOT 402 · abs 11101) · motorcycle · conf 0.69  
      ID 96 va ID 192 chong nhau IoU 0.69 trong 1 frame, tu frame 401  
      -> neu la 1 xe: xoa track thua (hoac merge); neu that su 2 xe che nhau: giu nguyen
- [ ] **frame 379** (MOT 380 · abs 11079) · motorcycle · conf 0.69  
      ID 199 va ID 214 chong nhau IoU 0.69 trong 2 frame, tu frame 379  
      -> neu la 1 xe: xoa track thua (hoac merge); neu that su 2 xe che nhau: giu nguyen
- [ ] **frame 236** (MOT 237 · abs 10936) · car · conf 0.68  
      ID 142 va ID 150 chong nhau IoU 0.68 trong 1 frame, tu frame 236  
      -> neu la 1 xe: xoa track thua (hoac merge); neu that su 2 xe che nhau: giu nguyen
- [ ] **frame 315** (MOT 316 · abs 11015) · motorcycle · conf 0.68  
      ID 96 va ID 182 chong nhau IoU 0.68 trong 1 frame, tu frame 315  
      -> neu la 1 xe: xoa track thua (hoac merge); neu that su 2 xe che nhau: giu nguyen
- [ ] **frame 428** (MOT 429 · abs 11128) · motorcycle · conf 0.68  
      ID 252 va ID 260 chong nhau IoU 0.68 trong 2 frame, tu frame 428  
      -> neu la 1 xe: xoa track thua (hoac merge); neu that su 2 xe che nhau: giu nguyen
- [ ] **frame 296** (MOT 297 · abs 10996) · motorcycle · conf 0.67  
      ID 158 va ID 176 chong nhau IoU 0.67 trong 1 frame, tu frame 296  
      -> neu la 1 xe: xoa track thua (hoac merge); neu that su 2 xe che nhau: giu nguyen
- [ ] **frame 582** (MOT 583 · abs 11282) · motorcycle · conf 0.67  
      ID 243 va ID 280 chong nhau IoU 0.67 trong 1 frame, tu frame 582  
      -> neu la 1 xe: xoa track thua (hoac merge); neu that su 2 xe che nhau: giu nguyen
- [ ] **frame 465** (MOT 466 · abs 11165) · motorcycle · conf 0.66  
      ID 227 va ID 229 chong nhau IoU 0.66 trong 1 frame, tu frame 465  
      -> neu la 1 xe: xoa track thua (hoac merge); neu that su 2 xe che nhau: giu nguyen
- [ ] **frame 299** (MOT 300 · abs 10999) · motorcycle · conf 0.66  
      ID 111 va ID 157 chong nhau IoU 0.66 trong 1 frame, tu frame 299  
      -> neu la 1 xe: xoa track thua (hoac merge); neu that su 2 xe che nhau: giu nguyen
- [ ] **frame 62** (MOT 63 · abs 10762) · motorcycle · conf 0.66  
      ID 30 va ID 56 chong nhau IoU 0.66 trong 1 frame, tu frame 62  
      -> neu la 1 xe: xoa track thua (hoac merge); neu that su 2 xe che nhau: giu nguyen

## GHOST — track qua ngan, nhieu kha nang la FP

Track song duoi 10 frame va khong cham mep hinh. Phan lon la detection rac; xoa nhanh, moi dong ~10 giay.


- [ ] **frame 236** (MOT 237 · abs 10936) · car · conf 0.90  
      ID 150 chi song 1 frame (236-236), box ~39px  
      -> neu khong co xe that o day -> xoa ca track (FP). Neu co xe -> keo dai track cho du
- [ ] **frame 6** (MOT 7 · abs 10706) · motorcycle · conf 0.80  
      ID 33 chi song 2 frame (6-7), box ~15px  
      -> neu khong co xe that o day -> xoa ca track (FP). Neu co xe -> keo dai track cho du
- [ ] **frame 62** (MOT 63 · abs 10762) · car · conf 0.80  
      ID 57 chi song 2 frame (62-63), box ~19px  
      -> neu khong co xe that o day -> xoa ca track (FP). Neu co xe -> keo dai track cho du
- [ ] **frame 139** (MOT 140 · abs 10839) · car · conf 0.80  
      ID 101 chi song 2 frame (139-140), box ~22px  
      -> neu khong co xe that o day -> xoa ca track (FP). Neu co xe -> keo dai track cho du
- [ ] **frame 598** (MOT 599 · abs 11298) · car · conf 0.80  
      ID 333 chi song 2 frame (598-599), box ~32px  
      -> neu khong co xe that o day -> xoa ca track (FP). Neu co xe -> keo dai track cho du
- [ ] **frame 598** (MOT 599 · abs 11298) · motorcycle · conf 0.80  
      ID 334 chi song 2 frame (598-599), box ~73px  
      -> neu khong co xe that o day -> xoa ca track (FP). Neu co xe -> keo dai track cho du
- [ ] **frame 598** (MOT 599 · abs 11298) · car · conf 0.80  
      ID 335 chi song 2 frame (598-599), box ~30px  
      -> neu khong co xe that o day -> xoa ca track (FP). Neu co xe -> keo dai track cho du
- [ ] **frame 573** (MOT 574 · abs 11273) · car · conf 0.60  
      ID 327 chi song 4 frame (573-576), box ~22px  
      -> neu khong co xe that o day -> xoa ca track (FP). Neu co xe -> keo dai track cho du
- [ ] **frame 596** (MOT 597 · abs 11296) · car · conf 0.60  
      ID 332 chi song 4 frame (596-599), box ~21px  
      -> neu khong co xe that o day -> xoa ca track (FP). Neu co xe -> keo dai track cho du
- [ ] **frame 181** (MOT 182 · abs 10881) · motorcycle · conf 0.50  
      ID 124 chi song 5 frame (181-185), box ~54px  
      -> neu khong co xe that o day -> xoa ca track (FP). Neu co xe -> keo dai track cho du
- [ ] **frame 261** (MOT 262 · abs 10961) · car · conf 0.50  
      ID 168 chi song 5 frame (261-265), box ~32px  
      -> neu khong co xe that o day -> xoa ca track (FP). Neu co xe -> keo dai track cho du
- [ ] **frame 349** (MOT 350 · abs 11049) · motorcycle · conf 0.30  
      ID 218 chi song 7 frame (349-355), box ~11px  
      -> neu khong co xe that o day -> xoa ca track (FP). Neu co xe -> keo dai track cho du
- [ ] **frame 461** (MOT 462 · abs 11161) · motorcycle · conf 0.30  
      ID 277 chi song 7 frame (461-467), box ~63px  
      -> neu khong co xe that o day -> xoa ca track (FP). Neu co xe -> keo dai track cho du
- [ ] **frame 62** (MOT 63 · abs 10762) · motorcycle · conf 0.20  
      ID 56 chi song 8 frame (62-69), box ~75px  
      -> neu khong co xe that o day -> xoa ca track (FP). Neu co xe -> keo dai track cho du
- [ ] **frame 418** (MOT 419 · abs 11118) · motorcycle · conf 0.10  
      ID 251 chi song 9 frame (418-426), box ~12px  
      -> neu khong co xe that o day -> xoa ca track (FP). Neu co xe -> keo dai track cho du

## ORPHAN — sinh/chet giua khung khong ghep duoc (kiem tra miss)

KHONG phai loi mac dinh: guide noi xe bi che han thi de track dut moi dung. Muc nay chi de kiem tra co phai xe bi miss hay khong. Lam sau cung, bo qua duoc neu het gio.


- [ ] **frame 276** (MOT 277 · abs 10976) · motorcycle · conf 0.87  
      ID 180 xuat hien giua khung o frame 276 (box 104px), khong tim thay track truoc do  
      -> tua lui truoc frame 276: neu xe da co tu truoc -> ve box va merge ve cung ID
- [ ] **frame 168** (MOT 169 · abs 10868) · motorcycle · conf 0.77  
      ID 54 bien mat giua khung o frame 168 (box 92px), khong tim thay track noi tiep  
      -> tua qua frame 168: neu xe bi che han -> de dut la dung; neu xe con thay -> ve tiep box
- [ ] **frame 66** (MOT 67 · abs 10766) · motorcycle · conf 0.75  
      ID 14 bien mat giua khung o frame 66 (box 90px), khong tim thay track noi tiep  
      -> tua qua frame 66: neu xe bi che han -> de dut la dung; neu xe con thay -> ve tiep box
- [ ] **frame 80** (MOT 81 · abs 10780) · motorcycle · conf 0.71  
      ID 48 bien mat giua khung o frame 80 (box 85px), khong tim thay track noi tiep  
      -> tua qua frame 80: neu xe bi che han -> de dut la dung; neu xe con thay -> ve tiep box
- [ ] **frame 295** (MOT 296 · abs 10995) · motorcycle · conf 0.70  
      ID 162 bien mat giua khung o frame 295 (box 84px), khong tim thay track noi tiep  
      -> tua qua frame 295: neu xe bi che han -> de dut la dung; neu xe con thay -> ve tiep box
- [ ] **frame 134** (MOT 135 · abs 10834) · motorcycle · conf 0.66  
      ID 43 bien mat giua khung o frame 134 (box 79px), khong tim thay track noi tiep  
      -> tua qua frame 134: neu xe bi che han -> de dut la dung; neu xe con thay -> ve tiep box
- [ ] **frame 458** (MOT 459 · abs 11158) · motorcycle · conf 0.62  
      ID 274 xuat hien giua khung o frame 458 (box 74px), khong tim thay track truoc do  
      -> tua lui truoc frame 458: neu xe da co tu truoc -> ve box va merge ve cung ID
- [ ] **frame 336** (MOT 337 · abs 11036) · motorcycle · conf 0.56  
      ID 156 bien mat giua khung o frame 336 (box 67px), khong tim thay track noi tiep  
      -> tua qua frame 336: neu xe bi che han -> de dut la dung; neu xe con thay -> ve tiep box
- [ ] **frame 240** (MOT 241 · abs 10940) · motorcycle · conf 0.56  
      ID 119 bien mat giua khung o frame 240 (box 67px), khong tim thay track noi tiep  
      -> tua qua frame 240: neu xe bi che han -> de dut la dung; neu xe con thay -> ve tiep box
- [ ] **frame 482** (MOT 483 · abs 11182) · motorcycle · conf 0.56  
      ID 289 xuat hien giua khung o frame 482 (box 67px), khong tim thay track truoc do  
      -> tua lui truoc frame 482: neu xe da co tu truoc -> ve box va merge ve cung ID
- [ ] **frame 52** (MOT 53 · abs 10752) · motorcycle · conf 0.55  
      ID 20 bien mat giua khung o frame 52 (box 67px), khong tim thay track noi tiep  
      -> tua qua frame 52: neu xe bi che han -> de dut la dung; neu xe con thay -> ve tiep box
- [ ] **frame 420** (MOT 421 · abs 11120) · motorcycle · conf 0.55  
      ID 217 bien mat giua khung o frame 420 (box 66px), khong tim thay track noi tiep  
      -> tua qua frame 420: neu xe bi che han -> de dut la dung; neu xe con thay -> ve tiep box
- [ ] **frame 491** (MOT 492 · abs 11191) · motorcycle · conf 0.54  
      ID 269 bien mat giua khung o frame 491 (box 64px), khong tim thay track noi tiep  
      -> tua qua frame 491: neu xe bi che han -> de dut la dung; neu xe con thay -> ve tiep box
- [ ] **frame 243** (MOT 244 · abs 10943) · car · conf 0.53  
      ID 13 bien mat giua khung o frame 243 (box 63px), khong tim thay track noi tiep  
      -> tua qua frame 243: neu xe bi che han -> de dut la dung; neu xe con thay -> ve tiep box
- [ ] **frame 299** (MOT 300 · abs 10999) · motorcycle · conf 0.51  
      ID 111 bien mat giua khung o frame 299 (box 61px), khong tim thay track noi tiep  
      -> tua qua frame 299: neu xe bi che han -> de dut la dung; neu xe con thay -> ve tiep box
- [ ] **frame 201** (MOT 202 · abs 10901) · motorcycle · conf 0.51  
      ID 68 bien mat giua khung o frame 201 (box 61px), khong tim thay track noi tiep  
      -> tua qua frame 201: neu xe bi che han -> de dut la dung; neu xe con thay -> ve tiep box
- [ ] **frame 402** (MOT 403 · abs 11102) · motorcycle · conf 0.50  
      ID 214 bien mat giua khung o frame 402 (box 60px), khong tim thay track noi tiep  
      -> tua qua frame 402: neu xe bi che han -> de dut la dung; neu xe con thay -> ve tiep box
- [ ] **frame 206** (MOT 207 · abs 10906) · motorcycle · conf 0.50  
      ID 86 bien mat giua khung o frame 206 (box 60px), khong tim thay track noi tiep  
      -> tua qua frame 206: neu xe bi che han -> de dut la dung; neu xe con thay -> ve tiep box
- [ ] **frame 389** (MOT 390 · abs 11089) · motorcycle · conf 0.48  
      ID 194 bien mat giua khung o frame 389 (box 58px), khong tim thay track noi tiep  
      -> tua qua frame 389: neu xe bi che han -> de dut la dung; neu xe con thay -> ve tiep box
- [ ] **frame 395** (MOT 396 · abs 11095) · motorcycle · conf 0.44  
      ID 235 xuat hien giua khung o frame 395 (box 53px), khong tim thay track truoc do  
      -> tua lui truoc frame 395: neu xe da co tu truoc -> ve box va merge ve cung ID
- [ ] **frame 219** (MOT 220 · abs 10919) · motorcycle · conf 0.44  
      ID 110 bien mat giua khung o frame 219 (box 53px), khong tim thay track noi tiep  
      -> tua qua frame 219: neu xe bi che han -> de dut la dung; neu xe con thay -> ve tiep box
- [ ] **frame 191** (MOT 192 · abs 10891) · motorcycle · conf 0.43  
      ID 128 xuat hien giua khung o frame 191 (box 52px), khong tim thay track truoc do  
      -> tua lui truoc frame 191: neu xe da co tu truoc -> ve box va merge ve cung ID
- [ ] **frame 318** (MOT 319 · abs 11018) · motorcycle · conf 0.42  
      ID 188 bien mat giua khung o frame 318 (box 50px), khong tim thay track noi tiep  
      -> tua qua frame 318: neu xe bi che han -> de dut la dung; neu xe con thay -> ve tiep box
- [ ] **frame 227** (MOT 228 · abs 10927) · motorcycle · conf 0.41  
      ID 146 xuat hien giua khung o frame 227 (box 49px), khong tim thay track truoc do  
      -> tua lui truoc frame 227: neu xe da co tu truoc -> ve box va merge ve cung ID
- [ ] **frame 381** (MOT 382 · abs 11081) · motorcycle · conf 0.41  
      ID 193 bien mat giua khung o frame 381 (box 49px), khong tim thay track noi tiep  
      -> tua qua frame 381: neu xe bi che han -> de dut la dung; neu xe con thay -> ve tiep box
- [ ] **frame 262** (MOT 263 · abs 10962) · motorcycle · conf 0.39  
      ID 169 xuat hien giua khung o frame 262 (box 46px), khong tim thay track truoc do  
      -> tua lui truoc frame 262: neu xe da co tu truoc -> ve box va merge ve cung ID
- [ ] **frame 66** (MOT 67 · abs 10766) · motorcycle · conf 0.37  
      ID 53 bien mat giua khung o frame 66 (box 44px), khong tim thay track noi tiep  
      -> tua qua frame 66: neu xe bi che han -> de dut la dung; neu xe con thay -> ve tiep box
- [ ] **frame 137** (MOT 138 · abs 10837) · motorcycle · conf 0.36  
      ID 100 xuat hien giua khung o frame 137 (box 43px), khong tim thay track truoc do  
      -> tua lui truoc frame 137: neu xe da co tu truoc -> ve box va merge ve cung ID
- [ ] **frame 449** (MOT 450 · abs 11149) · motorcycle · conf 0.35  
      ID 237 bien mat giua khung o frame 449 (box 43px), khong tim thay track noi tiep  
      -> tua qua frame 449: neu xe bi che han -> de dut la dung; neu xe con thay -> ve tiep box
- [ ] **frame 562** (MOT 563 · abs 11262) · motorcycle · conf 0.35  
      ID 299 bien mat giua khung o frame 562 (box 42px), khong tim thay track noi tiep  
      -> tua qua frame 562: neu xe bi che han -> de dut la dung; neu xe con thay -> ve tiep box
- [ ] **frame 584** (MOT 585 · abs 11284) · motorcycle · conf 0.34  
      ID 280 bien mat giua khung o frame 584 (box 41px), khong tim thay track noi tiep  
      -> tua qua frame 584: neu xe bi che han -> de dut la dung; neu xe con thay -> ve tiep box

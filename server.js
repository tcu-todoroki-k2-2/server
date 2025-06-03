// server.js

const fs = require('fs');
const path = require('path');
const ws = require('ws');
const cv = require('opencv4nodejs');
const YAML = require('yamljs');

// --------------------------------------------------------
// 1) ステレオキャリブレーション情報の読み込み
//    事前にキャリブレーションした YAML ファイルを置いておくこと
// --------------------------------------------------------
const stereoCalibYaml = YAML.load(path.resolve(__dirname, 'stereo_calib.yaml'));
// YAML の中身の例（OpenCV の stereoCalibrate で書き出したものを想定）：
// imageSize: [width, height]
// cameraMatrix1: [[fx1, 0, cx1], [0, fy1, cy1], [0, 0, 1]]
// distCoeffs1: [k1, k2, p1, p2, k3]
// cameraMatrix2: …
// distCoeffs2: …
// R: [[ … ], [ … ], [ … ]]      // 左→右 カメラへの回転行列
// T: [tx, ty, tz]                // 左→右 カメラ間の平行移動ベクトル
// ...

// カメラ行列や歪みパラメータを Mat に変換しておく
const cameraMatrix1 = new cv.Mat(stereoCalibYaml.cameraMatrix1, cv.CV_64F);
const distCoeffs1   = new cv.Mat(stereoCalibYaml.distCoeffs1, cv.CV_64F);
const cameraMatrix2 = new cv.Mat(stereoCalibYaml.cameraMatrix2, cv.CV_64F);
const distCoeffs2   = new cv.Mat(stereoCalibYaml.distCoeffs2, cv.CV_64F);
const R = new cv.Mat(stereoCalibYaml.R, cv.CV_64F);
const T = new cv.Mat([stereoCalibYaml.T], cv.CV_64F);

// ステレオ再投影行列 Q を計算しておく
// StereoRectify を呼び出し、Q を得る
const imageSize = new cv.Size(
  stereoCalibYaml.imageSize[0], 
  stereoCalibYaml.imageSize[1]
);
const { R1, R2, P1, P2, Q } = cv.stereoRectify(
  cameraMatrix1, distCoeffs1,
  cameraMatrix2, distCoeffs2,
  imageSize, R, T
);

// ステレオ BM もしくは SGBM の設定（見やすいように BM を使う例）
const numDisparities = 16 * 5;   // ここは環境に合わせて調整
const blockSize = 15;           // ブロックサイズも環境次第で変える
const stereoBM = new cv.StereoBM({
  numDisparities: numDisparities,
  blockSize: blockSize
});

// --------------------------------------------------------
// 2) クライアントごとのフレームバッファを用意
//    front 側、back 側から送られてきたフレームを一時的に保存して
//    タイムスタンプを合わせる
// --------------------------------------------------------
const frameBuffers = {
  front: [],
  back: []
};
// フレームバッファ要素の構造：{ seq, timestamp, matColor } 
//   seq       … クライアント側が送ってくる連番
//   timestamp … Date.now() 相当の ms（軽い揺らぎはある前提）
//   matColor  … JPEG → Mat(BGR) 変換した OpenCV Mat

// --------------------------------------------------------
// 3) 人物検出器 (HOG + SVM) を作成
// --------------------------------------------------------
const hog = new cv.HOGDescriptor();
hog.setSVMDetector(cv.HOGDescriptor.getDefaultPeopleDetector());

// --------------------------------------------------------
// 4) WebSocket サーバーを立てる
// --------------------------------------------------------
const wss = new ws.Server({ port: 8080 });
console.log('WebSocket サーバーをポート 8080 で起動');

// 接続してきたクライアント情報を保持（socket → role のマップ）
const clients = new Map();

wss.on('connection', (socket) => {
  let myRole = null; // このソケットが front/back のどちらか

  socket.on('message', async (data) => {
    // クライアントから来る JSON を parse
    let msg;
    try {
      msg = JSON.parse(data);
    } catch (e) {
      console.error('JSON parse error:', e);
      return;
    }

    // === (1) 役割情報 ===
    if (msg.type === 'role') {
      if (msg.role === 'front' || msg.role === 'back') {
        myRole = msg.role;
        clients.set(socket, myRole);
        console.log(`[サーバー] クライアントが接続: role=${myRole}`);
      }
      return;
    }

    // === (2) フレームデータ ===
    if (msg.type === 'frame' && (myRole === 'front' || myRole === 'back')) {
      // Base64 → Buffer → Mat (BGR)
      const imgBuf = Buffer.from(msg.img, 'base64');
      let matColor;
      try {
        matColor = cv.imdecode(imgBuf); // Mat (BGR) が得られる
      } catch (err) {
        console.warn('画像デコード失敗:', err);
        return;
      }

      // 現状は 320x240 の前提 → グレースケール化やリサイズは後で
      frameBuffers[myRole].push({
        seq: msg.seq,
        timestamp: msg.timestamp,
        matColor: matColor
      });

      // バッファ同士で timestamp が近いものを探してステレオ処理へ渡す
      attemptStereoPair();
    }
  });

  socket.on('close', () => {
    console.log(`[サーバー] クライアント切断: role=${myRole}`);
    clients.delete(socket);
  });
});

// --------------------------------------------------------
// 5) front/back のバッファから同期するペアを探し、あったら処理を実行
// --------------------------------------------------------
function attemptStereoPair() {
  if (frameBuffers.front.length === 0 || frameBuffers.back.length === 0) return;

  // 単純に一番古い front フレームと最も近い timestamp の back フレームをペアにする例
  const frontFrame = frameBuffers.front[0];
  let bestIdx = -1;
  let smallestDt = Number.MAX_SAFE_INTEGER;
  for (let i = 0; i < frameBuffers.back.length; i++) {
    const dt = Math.abs(frameBuffers.back[i].timestamp - frontFrame.timestamp);
    if (dt < smallestDt) {
      smallestDt = dt;
      bestIdx = i;
    }
  }
  // 時間差が大きすぎるならペアにしない（たとえば 200ms 以上はズレすぎ）
  if (smallestDt > 200) {
    // front 側が古すぎる可能性 → front を破棄
    frameBuffers.front.shift();
    return;
  }

  // ペアが見つかったら取り出す
  const backFrame = frameBuffers.back.splice(bestIdx, 1)[0];
  frameBuffers.front.shift();

  // ステレオ処理を非同期に実行
  processStereoPair(frontFrame.matColor, backFrame.matColor);
}

// --------------------------------------------------------
// 6) ステレオ三角測量＋人物検出して 3D 座標を計算
// --------------------------------------------------------
async function processStereoPair(imgLeftColor, imgRightColor) {
  try {
    // (A) グレースケール化 & リサイズ（320x240 → キャリブレーション画像サイズと合っていれば OK）
    const grayL = imgLeftColor.bgrToGray();
    const grayR = imgRightColor.bgrToGray();

    // (B) ステレオ BM で視差マップを計算
    //     → rectify（補正）していない画像ならば、まず stereoRectify で rectify してから。
    //     ここでは簡易例として、先ほど得た R1/R2, P1/P2 を使って rectify 画像を生成。
    const mapL = cv.initUndistortRectifyMap(
      cameraMatrix1, distCoeffs1, R1, P1, imageSize, cv.CV_16SC2
    );
    const mapR = cv.initUndistortRectifyMap(
      cameraMatrix2, distCoeffs2, R2, P2, imageSize, cv.CV_16SC2
    );
    const rectL = grayL.remap(mapL.x, mapL.y, cv.INTER_LINEAR);
    const rectR = grayR.remap(mapR.x, mapR.y, cv.INTER_LINEAR);

    const disp = stereoBM.compute(rectL, rectR);
    // disp は 16bit の Signed、各ピクセルの視差×16 の情報が格納されている

    // (C) 人検出 (HOG) をそれぞれの rectified カラー画像で実行
    //     → ここでは人 6 人前提。もし 6 人より多く検出されたら上位 6 つ、
    //        6 人より少なかったら検出された分だけ扱う。
    const rectLColor = imgLeftColor.remap(mapL.x, mapL.y, cv.INTER_LINEAR);
    const rectRColor = imgRightColor.remap(mapR.x, mapR.y, cv.INTER_LINEAR);
    const foundL = hog.detectMultiScale(rectLColor).objects; // 2D array of Rect
    const foundR = hog.detectMultiScale(rectRColor).objects;

    // (D) バウンディングボックス → 中心点 (x,y) を抽出
    const centersL = foundL.map(r => ({
      x: r.x + r.width/2,
      y: r.y + r.height/2
    }));
    const centersR = foundR.map(r => ({
      x: r.x + r.width/2,
      y: r.y + r.height/2
    }));

    // (E) 6 人分の対応付け (ここでは単純に検出順で対応させる)
    //     → 本当に厳密な対応付けが要るなら、顔認識 or tracker を併用したほうがいい
    const numDetected = Math.min(centersL.length, centersR.length, 6);
    const result3D = [];
    for (let i = 0; i < numDetected; i++) {
      const ptL = centersL[i];
      const ptR = centersR[i];
      // 視差 d = xL - xR
      const d = ptL.x - ptR.x;
      if (d <= 0) {
        // 視差が 0 以下だと三角測量できない → 無視して奥に置いておく
        result3D.push({ x: 0, y: 0, z: -1 });
      } else {
        // 3D 再投影行列 Q を使って (x,y,d) から (X,Y,Z) を得る方法もあるが、
        // ここでは f, B (ベースライン) を YAML で与えておいて手動計算してもOK
        //  → 例： X = (xL - cx) * Z / fx, Y = (yL - cy) * Z / fy, Z = fx * B / d
        //    (cx, cy) は cameraMatrix1 から読み取れる。B は T の x 成分 (平行配置の場合)。
        const fx = cameraMatrix1.at(0, 0);
        const fy = cameraMatrix1.at(1, 1);
        const cx = cameraMatrix1.at(0, 2);
        const cy = cameraMatrix1.at(1, 2);
        // 両カメラ間のベースライン B は T の x 成分 (mm 単位なら注意)
        const B = Math.abs(T.at(0, 0)); // 実際にはメートル単位で換算する必要がある
        const Z = fx * B / d;          // 深度
        const X = (ptL.x - cx) * Z / fx;
        const Y = (ptL.y - cy) * Z / fy;
        result3D.push({ x: X, y: Y, z: Z });
      }
    }

    // もし６人より少なければ、残りは (0,0,-1) のダミー
    for (let k = result3D.length; k < 6; k++) {
      result3D.push({ x: 0, y: 0, z: -1 });
    }

    // (F) すべての接続中クライアントに 3D 座標を broadcast
    const payload = JSON.stringify({ type: '3d_positions', data: result3D });
    for (const [cli, r] of clients.entries()) {
      if (cli.readyState === ws.OPEN) {
        cli.send(payload);
      }
    }
  } catch (err) {
    console.error('stereo processing error:', err);
  }
}

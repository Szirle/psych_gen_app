import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:psych_gen_app/features/face_generation/domain/entities/full_resolution_face.dart';
import 'package:psych_gen_app/features/face_generation/presentation/widgets/face_generation/quick_look_image_dialog.dart';

Uint8List _pixelBytes() => base64Decode(
      'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=',
    );

void main() {
  testWidgets('arrow keys navigate by grid axis without cycling',
      (tester) async {
    final pixel = _pixelBytes();
    final cached = FullResolutionFace(
      imageBytes: pixel,
      latentNpyBytes: Uint8List(18 * 512 * 4),
    );
    final previews = <List<int>>[];
    final requested = <List<int>>[];
    final unresolved = Completer<FullResolutionFace>();

    await tester.pumpWidget(
      MaterialApp(
        home: QuickLookImageDialog(
          previewBytes: pixel,
          fullResolutionFace: cached,
          initialSelectedLevels: const [0, 0],
          levelCounts: const [3, 2],
          previewForLevels: (levels) {
            previews.add(List<int>.from(levels));
            return pixel;
          },
          cachedFullResolutionForLevels: (levels) =>
              levels[0] == 1 && levels[1] == 0 ? cached : null,
          requestFullResolutionForLevels: (levels, preview) {
            requested.add(List<int>.from(levels));
            return unresolved.future;
          },
        ),
      ),
    );
    await tester.pump();

    await tester.sendKeyEvent(LogicalKeyboardKey.arrowLeft);
    expect(previews, isEmpty);

    await tester.sendKeyEvent(LogicalKeyboardKey.arrowRight);
    await tester.sendKeyEvent(LogicalKeyboardKey.arrowUp);

    expect(previews, const [
      <int>[1, 0],
      <int>[1, 1]
    ]);
    expect(requested, const [
      <int>[1, 1]
    ]);
    expect(find.byTooltip('Download Image'), findsOneWidget);
    expect(find.byTooltip('Download Latent'), findsOneWidget);
  });
}

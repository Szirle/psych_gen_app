import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/gestures.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:psych_gen_app/core/designsystem/widgets/shimmer_image_placeholder.dart';

Uint8List _pixelBytes() => base64Decode(
      'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=',
    );

Widget _canvasHarness({
  required TransformationController controller,
  required VoidCallback onTap,
}) {
  return MaterialApp(
    home: Center(
      child: SizedBox.square(
        dimension: 320,
        child: InteractiveViewer(
          transformationController: controller,
          panEnabled: true,
          scaleEnabled: true,
          panAxis: PanAxis.free,
          minScale: 0.5,
          maxScale: 6,
          boundaryMargin: const EdgeInsets.all(1000),
          trackpadScrollCausesScale: true,
          scaleFactor: 160,
          child: Center(
            child: AsyncGeneratedImageTile(
              imageBytes: _pixelBytes(),
              size: 100,
              isLoading: false,
              onTap: onTap,
            ),
          ),
        ),
      ),
    ),
  );
}

void main() {
  testWidgets('a short image click opens the image', (tester) async {
    final controller = TransformationController();
    var taps = 0;
    await tester.pumpWidget(
      _canvasHarness(controller: controller, onTap: () => taps++),
    );

    await tester.tap(find.byType(AsyncGeneratedImageTile));

    expect(taps, 1);
    controller.dispose();
  });

  testWidgets('dragging from an image pans without opening it', (tester) async {
    final controller = TransformationController();
    var taps = 0;
    await tester.pumpWidget(
      _canvasHarness(controller: controller, onTap: () => taps++),
    );

    await tester.drag(
      find.byType(AsyncGeneratedImageTile),
      const Offset(70, 45),
    );
    await tester.pumpAndSettle();

    expect(taps, 0);
    expect(controller.value.getTranslation().x.abs(), greaterThan(0));
    expect(controller.value.getTranslation().y.abs(), greaterThan(0));
    controller.dispose();
  });

  testWidgets('mouse wheel zooms the canvas around the pointer',
      (tester) async {
    final controller = TransformationController();
    await tester.pumpWidget(
      _canvasHarness(controller: controller, onTap: () {}),
    );

    await tester.sendEventToBinding(
      PointerScrollEvent(
        position: tester.getCenter(find.byType(InteractiveViewer)),
        scrollDelta: const Offset(0, -120),
      ),
    );
    await tester.pump();

    expect(controller.value.getMaxScaleOnAxis(), greaterThan(1));
    controller.dispose();
  });
}

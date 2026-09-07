import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:psych_gen_app/core/designsystem/widgets/custom_number_text_field.dart';
import 'package:psych_gen_app/core/designsystem/widgets/safe_memory_image.dart';
import 'package:psych_gen_app/core/designsystem/widgets/shimmer_image_placeholder.dart';
import 'package:psych_gen_app/features/face_generation/data/datasources/face_manipulation_api_datasource.dart';
import 'package:psych_gen_app/features/face_generation/domain/entities/manipulated_dimension.dart';
import 'package:psych_gen_app/features/face_generation/domain/entities/manipulated_dimension_name.dart';
import 'package:psych_gen_app/features/face_generation/presentation/bloc/face_manipulation_state.dart';
import 'package:psych_gen_app/features/face_generation/presentation/widgets/face_generation/preview_grid_coordinates.dart';
import 'package:shimmer/shimmer.dart';

String _encoded(int value) => base64Encode(Uint8List.fromList([value]));

Uint8List _pixelBytes() => base64Decode(
      'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=',
    );

List<int> _firstBytes(List<Uint8List> images) =>
    images.map((image) => image.first).toList();

void main() {
  test('retains existing endpoint faces and leaves a new middle level empty',
      () {
    final grid = FaceImageGrid(
      images: [
        Uint8List.fromList([1]),
        Uint8List.fromList([2])
      ],
      dimensionNames: const ['dominant'],
      levelValues: const [
        [-25, 25],
      ],
    );
    final target = [
      ManipulatedDimension(
        name: ManipulatedDimensionName.dominant,
        strength: 25,
        nLevels: 3,
      ),
    ];

    expect(grid.imageFor(target, [0])?.first, 1);
    expect(grid.imageFor(target, [1]), isNull);
    expect(grid.imageFor(target, [2])?.first, 2);
  });

  testWidgets('loading shimmer overlays the retained image', (tester) async {
    await tester.pumpWidget(
      MaterialApp(
        home: Center(
          child: AsyncGeneratedImageTile(
            imageBytes: _pixelBytes(),
            size: 100,
            isLoading: true,
          ),
        ),
      ),
    );

    expect(find.byType(SafeMemoryImage), findsOneWidget);
    expect(find.byType(Shimmer), findsOneWidget);
  });

  testWidgets(
      'replacement keeps outgoing and incoming images in one fixed tile',
      (tester) async {
    Widget tile(Uint8List bytes) => MaterialApp(
          home: Center(
            child: AsyncGeneratedImageTile(
              key: const ValueKey('tile'),
              imageBytes: bytes,
              size: 100,
              isLoading: false,
            ),
          ),
        );

    final outgoing = _pixelBytes();
    final incoming = Uint8List.fromList([...outgoing, 0]);
    await tester.pumpWidget(tile(outgoing));
    await tester.pumpWidget(tile(incoming));

    expect(find.byType(SafeMemoryImage), findsNWidgets(2));
    expect(find.byType(Opacity), findsNothing);
    expect(
      tester.getSize(find.byType(AsyncGeneratedImageTile)),
      const Size(100, 100),
    );
  });

  test('decodes an unequal 2D response with dimension zero changing fastest',
      () {
    final grid = <dynamic>[
      <dynamic>[_encoded(0), _encoded(1), _encoded(2)],
      <dynamic>[_encoded(10), _encoded(11), _encoded(12)],
      <dynamic>[_encoded(20), _encoded(21), _encoded(22)],
      <dynamic>[_encoded(30), _encoded(31), _encoded(32)],
    ];

    expect(
      _firstBytes(decodeImageGridForDisplay(grid, [3, 4])),
      [0, 1, 2, 10, 11, 12, 20, 21, 22, 30, 31, 32],
    );
  });

  test('decodes an unequal 3D response with depth changing slowest', () {
    final grid = <dynamic>[
      <dynamic>[
        <dynamic>[_encoded(0), _encoded(100)],
        <dynamic>[_encoded(10), _encoded(110)],
        <dynamic>[_encoded(20), _encoded(120)],
      ],
      <dynamic>[
        <dynamic>[_encoded(1), _encoded(101)],
        <dynamic>[_encoded(11), _encoded(111)],
        <dynamic>[_encoded(21), _encoded(121)],
      ],
    ];

    expect(
      _firstBytes(decodeImageGridForDisplay(grid, [2, 3, 2])),
      [0, 1, 10, 11, 20, 21, 100, 101, 110, 111, 120, 121],
    );
  });

  test('vertical display rows map from high values at top to low at bottom',
      () {
    final grid = FaceImageGrid(
      images: List.generate(12, (index) => Uint8List.fromList([index])),
      dimensionNames: const ['dominant', 'well-groomed'],
      levelValues: const [
        [-1, 0, 1],
        [-2, -0.67, 0.67, 2],
      ],
    );
    final dimensions = [
      ManipulatedDimension(
        name: ManipulatedDimensionName.dominant,
        strength: 1,
        nLevels: 3,
      ),
      ManipulatedDimension(
        name: ManipulatedDimensionName.wellGroomed,
        strength: 2,
        nLevels: 4,
      ),
    ];

    final topLevel = verticalLevelForDisplayRow(0, 4);
    final bottomLevel = verticalLevelForDisplayRow(3, 4);

    expect(topLevel, 3);
    expect(bottomLevel, 0);
    expect(grid.imageFor(dimensions, [0, topLevel])?.first, 9);
    expect(grid.imageFor(dimensions, [0, bottomLevel])?.first, 0);
    // Horizontal coordinates remain low-to-high from left to right.
    expect(grid.imageFor(dimensions, [2, topLevel])?.first, 11);
  });

  testWidgets('number field arrows notify listeners', (tester) async {
    int? changedValue;
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: CustomNumberTextField(
            onChanged: (value) => changedValue = value,
          ),
        ),
      ),
    );

    await tester.enterText(find.byType(TextField), '3');
    await tester.tap(find.byIcon(Icons.arrow_drop_up));
    await tester.pump();

    expect(changedValue, 4);
  });
}

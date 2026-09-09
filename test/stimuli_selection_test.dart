import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:easy_localization/easy_localization.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:psych_gen_app/core/designsystem/widgets/shimmer_image_placeholder.dart';
import 'package:psych_gen_app/features/face_generation/data/datasources/face_manipulation_api_datasource.dart';
import 'package:psych_gen_app/features/face_generation/data/datasources/stimuli_selection_api_datasource.dart';
import 'package:psych_gen_app/features/face_generation/domain/entities/face_manipulation_request.dart';
import 'package:psych_gen_app/features/face_generation/domain/entities/manipulated_dimension_name.dart';
import 'package:psych_gen_app/features/face_generation/domain/entities/stimuli_selection_request.dart';
import 'package:psych_gen_app/features/face_generation/domain/repositories/face_manipulation_repository.dart';
import 'package:psych_gen_app/features/face_generation/domain/repositories/distributions_repository.dart';
import 'package:psych_gen_app/features/face_generation/domain/usecases/generate_face_images.dart';
import 'package:psych_gen_app/features/face_generation/domain/usecases/fetch_distributions.dart';
import 'package:psych_gen_app/features/face_generation/presentation/bloc/face_manipulation_bloc.dart';
import 'package:psych_gen_app/features/face_generation/presentation/bloc/filters_bloc.dart';
import 'package:psych_gen_app/features/face_generation/presentation/bloc/stimuli_selection_cubit.dart';
import 'package:psych_gen_app/features/face_generation/presentation/pages/face_generation_page.dart';
import 'package:psych_gen_app/features/face_generation/presentation/widgets/filters_panel.dart';
import 'package:psych_gen_app/features/face_generation/presentation/widgets/face_generation/settings_panel.dart';

const pixel =
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=';
StimuliSelectionRequest selection(int revision,
        {Map<ManipulatedDimensionName, List<double>> filters = const {}}) =>
    StimuliSelectionRequest(
        previewRevision: revision,
        truncationPsi: .6,
        selection: StimuliSelectionCriteria(filters: filters));
Map<String, dynamic> responseBody(int revision) => {
      'preview_revision': revision,
      'images': List.generate(3, (_) => List.filled(3, pixel)),
      'sampling': {
        'source': 'generator',
        'eligible_count': null,
        'sampled_with_replacement': false
      }
    };
StimuliSelectionPreview preview(int revision) => StimuliSelectionPreview(
    images: List.generate(9, (_) => base64Decode(pixel)),
    previewRevision: revision,
    source: 'generator',
    eligibleCount: null,
    sampledWithReplacement: false);

class FaceRepository implements FaceManipulationRepository {
  final requests = <FaceManipulationRequest>[];
  @override
  Future<List<Uint8List>> getFaceImages(FaceManipulationRequest request) async {
    requests.add(request);
    return List.generate(2, (_) => base64Decode(pixel));
  }
}

class HistogramRepository implements DistributionsRepository {
  final pending = Completer<Map<ManipulatedDimensionName, List<double>>>();
  @override
  Future<Map<ManipulatedDimensionName, List<double>>> fetchDistributions(
          {Map<ManipulatedDimensionName, List<double>>? filters,
          int numPoints = 100,
          List<ManipulatedDimensionName>? variables}) =>
      pending.future;
}

class FaceApi extends FaceManipulationApiDataSource {
  final cancellations = <int>[];
  @override
  Future<void> cancelFullResolutionGeneration(int revision) async {
    cancellations.add(revision);
  }
}

void main() {
  final logLevels = EasyLocalization.logger.enableLevels;
  setUp(() => EasyLocalization.logger.enableLevels = []);
  tearDown(() => EasyLocalization.logger.enableLevels = logLevels);
  test('request snapshots sampling criteria independently of manipulation', () {
    final ranges = {
      ManipulatedDimensionName.dominant: [.2, .8]
    };
    final request = selection(10, filters: ranges);
    ranges.values.first[0] = .9;
    ranges.clear();
    expect(request.toJson()['selection'], {
      'filters': {
        'dominant': [.2, .8]
      }
    });
    expect(request.toJson().containsKey('manipulated_dimensions'), isFalse);
    expect(() => request.selection.filters.values.first[0] = .5,
        throwsUnsupportedError);
  });
  test('transport validates revision, shape and empty-population errors',
      () async {
    var variant = 0;
    final api = StimuliSelectionApiDataSource(
        client: MockClient((http.Request request) async {
      expect(request.url.path, '/stimuli/preview');
      final body = jsonDecode(request.body);
      expect(body['sample_count'], 9);
      expect(body['selection']['filters'], {
        'wellGroomed': [.2, .8]
      });
      if (variant == 3) {
        return http.Response('{"error":"No eligible faces"}', 422);
      }
      final response = responseBody(variant == 1 ? 6 : 5);
      if (variant == 2) {
        response['images'] = [
          [pixel]
        ];
      }
      return http.Response(jsonEncode(response), 200);
    }));
    final request = selection(5, filters: {
      ManipulatedDimensionName.wellGroomed: [.2, .8]
    });
    expect((await api.fetchPreview(request)).images.length, 9);
    for (variant = 1; variant < 3; variant++) {
      await expectLater(api.fetchPreview(request), throwsFormatException);
    }
    await expectLater(api.fetchPreview(request),
        throwsA(predicate((e) => e.toString().contains('No eligible faces'))));
    api.close();
  });
  testWidgets('latest result wins and failure clears invalid previous sample',
      (tester) async {
    final calls = List.generate(4, (_) => Completer<StimuliSelectionPreview>());
    var index = 0;
    final cubit = StimuliSelectionCubit(
        debounce: Duration.zero, fetchPreview: (_) => calls[index++].future);
    cubit.load(selection(1));
    await tester.pump(const Duration(milliseconds: 1));
    cubit.load(selection(2));
    await tester.pump(const Duration(milliseconds: 1));
    calls[1].complete(preview(2));
    await tester.pump();
    calls[0].complete(preview(1));
    await tester.pump();
    expect(cubit.state.preview?.previewRevision, 2);
    cubit.load(selection(3));
    await tester.pump(const Duration(milliseconds: 1));
    expect(cubit.state.preview?.previewRevision, 2);
    expect(cubit.state.isLoading, isTrue);
    calls[2].completeError(Exception('empty population'));
    await tester.pump();
    expect(cubit.state.preview, isNull);
    expect(cubit.state.error, contains('empty population'));
    cubit.load(selection(4));
    await tester.pump(const Duration(milliseconds: 1));
    cubit.cancelPending();
    calls[3].complete(preview(4));
    await tester.pump();
    expect(cubit.state.preview, isNull);
    await cubit.close();
  });
  testWidgets(
      'tabs separate settings and preserve committed filters during histogram loads',
      (tester) async {
    tester.view.physicalSize = const Size(1200, 1000);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);
    final selectionRequests = <Map>[];
    final api = StimuliSelectionApiDataSource(
        client: MockClient((http.Request request) async {
      final body = jsonDecode(request.body) as Map;
      selectionRequests.add(body);
      return http.Response(
          jsonEncode(responseBody(body['preview_revision'])), 200);
    }));
    final repo = FaceRepository();
    final face = FaceManipulationBloc(
        generateFaceImages: GenerateFaceImagesUseCase(repository: repo));
    final histograms = HistogramRepository();
    final filters = FiltersBloc(
        fetchDistributions: FetchDistributionsUseCase(repository: histograms));
    final faceApi = FaceApi();
    await tester.pumpWidget(MaterialApp(
        home: MultiBlocProvider(
            providers: [
          BlocProvider.value(value: face),
          BlocProvider.value(value: filters)
        ],
            child: FaceGenerationPage(
                title: 'PsychGan', selectionApi: api, faceApi: faceApi))));
    await tester.pump(const Duration(milliseconds: 600));
    await tester.pump();
    expect(find.byType(FiltersPanel), findsOneWidget);
    expect(find.byType(SettingsPanel), findsNothing);
    expect(find.byType(AsyncGeneratedImageTile), findsNWidgets(9));
    expect(repo.requests, isEmpty);
    expect(selectionRequests.single['selection']['filters'], isEmpty);
    tester.widget<FiltersPanel>(find.byType(FiltersPanel)).onFiltersCommitted({
      ManipulatedDimensionName.dominant: [.2, .7]
    });
    await tester.pump();
    expect(filters.state, isA<FiltersLoading>());
    await tester.tap(find.byType(Tab).at(1));
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 600));
    await tester.pump();
    expect(find.byType(FiltersPanel), findsNothing);
    expect(find.byType(SettingsPanel), findsOneWidget);
    expect(
        tester.widget<SettingsPanel>(find.byType(SettingsPanel)).showTruncation,
        isFalse);
    expect(repo.requests.single.filters, {
      ManipulatedDimensionName.dominant: [.2, .7]
    });
    expect(repo.requests.single.changeFace, isFalse);
    await tester.tap(find.byType(Tab).first);
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 600));
    await tester.pump();
    expect(find.byType(AsyncGeneratedImageTile), findsNWidgets(9));
    expect(selectionRequests.last['selection']['filters'], {
      'dominant': [.2, .7]
    });
    expect(repo.requests.length, 1);
    await tester.pumpWidget(const SizedBox());
    histograms.pending.complete({});
    // Match BlocProvider disposal: initiate close without awaiting broadcast
    // stream shutdown inside the widget test's fake-async zone.
    unawaited(face.close());
    unawaited(filters.close());
    await tester.pump(const Duration(seconds: 1));
  });
}

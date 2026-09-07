import 'package:easy_localization/easy_localization.dart';
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:psych_gen_app/features/face_generation/data/repositories/distributions_repository_impl.dart';
import 'package:psych_gen_app/features/face_generation/data/repositories/face_manipulation_repository_impl.dart';
import 'package:psych_gen_app/features/face_generation/domain/usecases/fetch_distributions.dart';
import 'package:psych_gen_app/features/face_generation/domain/usecases/generate_face_images.dart';
import 'package:psych_gen_app/features/face_generation/presentation/bloc/face_manipulation_bloc.dart';
import 'package:psych_gen_app/features/face_generation/presentation/bloc/filters_bloc.dart';
import 'package:psych_gen_app/features/face_generation/presentation/pages/face_generation_page.dart';
import 'package:psych_gen_app/core/designsystem/app_theme.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  EasyLocalization.ensureInitialized().then((_) {
    runApp(EasyLocalization(
      supportedLocales: const [Locale('en')],
      path: 'assets/translations',
      fallbackLocale: const Locale('en'),
      child: const TraversalsApp(),
    ));
  });
}

class TraversalsApp extends StatefulWidget {
  const TraversalsApp({super.key});

  @override
  State<TraversalsApp> createState() => _TraversalsAppState();
}

class _TraversalsAppState extends State<TraversalsApp> {
  ThemeMode _themeMode = ThemeMode.system;

  @override
  Widget build(BuildContext context) => MaterialApp(
        title: 'Traversal explorer',
        debugShowCheckedModeBanner: false,
        localizationsDelegates: context.localizationDelegates,
        supportedLocales: context.supportedLocales,
        locale: context.locale,
        theme: AppTheme.light(),
        darkTheme: AppTheme.dark(),
        themeMode: _themeMode,
        home: MultiBlocProvider(
          providers: [
            BlocProvider(
              create: (_) => FaceManipulationBloc(
                generateFaceImages: GenerateFaceImagesUseCase(
                  repository: FaceManipulationRepositoryImpl(),
                ),
              ),
            ),
            BlocProvider(
              create: (_) => FiltersBloc(
                fetchDistributions: FetchDistributionsUseCase(
                  repository: DistributionsRepositoryImpl(),
                ),
              ),
            ),
          ],
          child: FaceGenerationPage(
            title: 'Traversal explorer',
            traversalMode: true,
            numTraversals: 512,
            onThemeModeChanged: (isDark) {
              setState(() {
                _themeMode = isDark ? ThemeMode.dark : ThemeMode.light;
              });
            },
          ),
        ),
      );
}

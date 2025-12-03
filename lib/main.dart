import 'package:flutter/material.dart';
import 'package:easy_localization/easy_localization.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:psych_gen_app/features/face_generation/presentation/bloc/face_manipulation_bloc.dart';
import 'package:psych_gen_app/features/face_generation/presentation/pages/face_generation_page.dart';
import 'package:psych_gen_app/features/face_generation/domain/usecases/generate_face_images.dart';
import 'package:psych_gen_app/features/face_generation/data/repositories/face_manipulation_repository_impl.dart';
import 'package:psych_gen_app/features/face_generation/presentation/bloc/filters_bloc.dart';
import 'package:psych_gen_app/features/face_generation/domain/usecases/fetch_distributions.dart';
import 'package:psych_gen_app/features/face_generation/data/repositories/distributions_repository_impl.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  EasyLocalization.ensureInitialized().then((_) {
    runApp(
      EasyLocalization(
        supportedLocales: const [Locale('en')],
        path: 'assets/translations',
        fallbackLocale: const Locale('en'),
        child: const MyApp(),
      ),
    );
  });
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'app.title'.tr(),
      debugShowCheckedModeBanner: false,
      localizationsDelegates: context.localizationDelegates,
      supportedLocales: context.supportedLocales,
      locale: context.locale,
      theme: ThemeData(
        fontFamily: 'WorkSans',
        scaffoldBackgroundColor: Colors.grey.shade50,
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.black),
        useMaterial3: true,
      ),
      home: MultiBlocProvider(
        providers: [
          BlocProvider(
            create: (context) => FaceManipulationBloc(
              generateFaceImages: GenerateFaceImagesUseCase(
                repository: FaceManipulationRepositoryImpl(),
              ),
            ),
          ),
          BlocProvider(
            create: (context) => FiltersBloc(
              fetchDistributions: FetchDistributionsUseCase(
                repository: DistributionsRepositoryImpl(),
              ),
            ),
          ),
        ],
        child: FaceGenerationPage(title: 'app.title'.tr()),
      ),
    );
  }
}

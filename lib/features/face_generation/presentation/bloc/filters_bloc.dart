import 'package:bloc/bloc.dart';
import 'package:equatable/equatable.dart';
import 'package:psych_gen_app/features/face_generation/domain/entities/manipulated_dimension_name.dart';
import 'package:psych_gen_app/features/face_generation/domain/usecases/fetch_distributions.dart';

part 'filters_event.dart';
part 'filters_state.dart';

class FiltersBloc extends Bloc<FiltersEvent, FiltersState> {
  final FetchDistributionsUseCase fetchDistributions;

  FiltersBloc({required this.fetchDistributions}) : super(FiltersInitial()) {
    on<LoadDistributionsEvent>(_onLoad);
    on<UpdateFilterEvent>(_onUpdateFilter);
    on<CommitFilterEvent>(_onCommitFilter);
  }

  Map<ManipulatedDimensionName, List<double>> _filters = {};
  int _latestLoad = 0;

  Map<ManipulatedDimensionName, List<double>> _snapshotFilters() =>
      _filters.map((key, value) => MapEntry(key, List<double>.from(value)));

  Future<void> _onLoad(
      LoadDistributionsEvent event, Emitter<FiltersState> emit) async {
    emit(FiltersLoading());
    final loadId = ++_latestLoad;
    final filters = _snapshotFilters();
    try {
      final data = await fetchDistributions(
        filters: filters,
        numPoints: event.numPoints,
        variables: event.variables,
      );
      if (loadId != _latestLoad || emit.isDone) return;
      emit(FiltersLoaded(distributions: data, appliedFilters: filters));
    } catch (e) {
      if (loadId != _latestLoad || emit.isDone) return;
      emit(FiltersError(message: e.toString()));
    }
  }

  Future<void> _onUpdateFilter(
      UpdateFilterEvent event, Emitter<FiltersState> emit) async {
    _filters = Map<ManipulatedDimensionName, List<double>>.from(_filters);
    if (event.range == null) {
      _filters.remove(event.dimension);
    } else {
      _filters[event.dimension] = List<double>.from(event.range!);
    }
    // Do NOT auto-reload distributions on every change; update state locally.
    final current = state;
    if (current is FiltersLoaded) {
      emit(FiltersLoaded(
        distributions: current.distributions,
        appliedFilters: _snapshotFilters(),
      ));
    }
  }

  Future<void> _onCommitFilter(
      CommitFilterEvent event, Emitter<FiltersState> emit) async {
    _filters = _snapshotFilters();
    if (event.range == null) {
      _filters.remove(event.dimension);
    } else {
      _filters[event.dimension] = List<double>.from(event.range!);
    }
    final filters = _snapshotFilters();
    final loadId = ++_latestLoad;
    emit(FiltersLoading());
    try {
      final data = await fetchDistributions(
        filters: filters,
        numPoints: event.numPoints,
        variables: event.variables,
      );
      if (loadId != _latestLoad || emit.isDone) return;
      emit(FiltersLoaded(distributions: data, appliedFilters: filters));
    } catch (e) {
      if (loadId != _latestLoad || emit.isDone) return;
      emit(FiltersError(message: e.toString()));
    }
  }
}

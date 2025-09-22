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
  }

  Map<ManipulatedDimensionName, List<double>> _filters = {};

  Future<void> _onLoad(
      LoadDistributionsEvent event, Emitter<FiltersState> emit) async {
    emit(FiltersLoading());
    try {
      final data = await fetchDistributions(
        filters: _filters,
        numPoints: event.numPoints,
        variables: event.variables,
      );
      emit(FiltersLoaded(distributions: data, appliedFilters: _filters));
    } catch (e) {
      emit(FiltersError(message: e.toString()));
    }
  }

  Future<void> _onUpdateFilter(
      UpdateFilterEvent event, Emitter<FiltersState> emit) async {
    _filters = Map<ManipulatedDimensionName, List<double>>.from(_filters);
    if (event.range == null) {
      _filters.remove(event.dimension);
    } else {
      _filters[event.dimension] = event.range!;
    }
    // Do NOT auto-reload distributions on every change; update state locally.
    final current = state;
    if (current is FiltersLoaded) {
      emit(FiltersLoaded(
          distributions: current.distributions, appliedFilters: _filters));
    }
  }
}

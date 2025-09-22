part of 'filters_bloc.dart';

abstract class FiltersState extends Equatable {
  const FiltersState();
  @override
  List<Object?> get props => [];
}

class FiltersInitial extends FiltersState {}

class FiltersLoading extends FiltersState {}

class FiltersLoaded extends FiltersState {
  final Map<ManipulatedDimensionName, List<double>> distributions;
  final Map<ManipulatedDimensionName, List<double>> appliedFilters;
  const FiltersLoaded(
      {required this.distributions, required this.appliedFilters});
  @override
  List<Object?> get props => [distributions, appliedFilters];
}

class FiltersError extends FiltersState {
  final String message;
  const FiltersError({required this.message});
  @override
  List<Object?> get props => [message];
}
